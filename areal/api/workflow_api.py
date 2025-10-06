import asyncio
import queue
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch.distributed as dist
import uvloop
from megatron.core import parallel_state as mpu
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import RolloutStat
from areal.experimental.openai.types import CompletionWithTokenLogpReward
from areal.utils import logging
from areal.utils.data import concat_padded_tensors, cycle_dataloader

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
import uuid
worker_id = uuid.uuid4().hex[:4]
logger = logging.getLogger(f"areal.workflow_api @ {worker_id}")


ROLLOUT_POLL_WAIT_TIME = 0.05


class RolloutWorkflow:

    async def arun_episode(
        self, engine: "InferenceEngine", data: Dict[str, Any]
    ) -> Union[TensorDict, None, Dict[str, CompletionWithTokenLogpReward]]:
        """Run a single episode of the workflow.

        `None` implies that this trajectory is rejected and will not be used for training.

        See concrete example implementations under the `areal/workflow` directory.
        """
        raise NotImplementedError()


@dataclass
class _TimedResult:
    t: int
    data: TensorDict


@dataclass
class _RolloutTaskInput:
    data: Dict[str, Any]
    workflow: RolloutWorkflow
    should_accept: Callable | None = None


@dataclass
class _RolloutTask:
    create_time: int
    task: asyncio.Task
    task_input: _RolloutTaskInput


class WorkflowExecutor:

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: "InferenceEngine",
    ):
        self.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config
        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()

        self.inference_engine = inference_engine

        qsize = config.queue_size or self.max_concurrent_rollouts * 16
        self.input_queue = queue.Queue(maxsize=qsize)
        self.output_queue = queue.Queue(maxsize=qsize)
        self.result_cache: List[_TimedResult] = []

        self.rollout_stat = RolloutStat()
        
        # Thread pool configuration
        self.max_rollout_threads = getattr(config, 'max_rollout_threads', 8)
        self.rollout_executor = None
        self.coordinator_thread = None
        
        # Thread-safe task tracking
        self.thread_task_counts = {}  # thread_id -> current_task_count
        self.thread_task_lock = threading.Lock()
        self.task_assignment_queue = queue.Queue()  # For assigning tasks to threads
        self.task_completion_queue = queue.Queue()  # For collecting completed tasks

    def initialize(self, logger=None, train_data_parallel_size: int | None = None):
        if logger is None:
            logger = logging.getLogger("WorkflowExecutor")
        self.logger = logger

        if train_data_parallel_size is not None:
            self.dp_world_size = train_data_parallel_size
        else:
            if dist.is_initialized():
                if not mpu.is_initialized():
                    self.dp_world_size = dist.get_world_size()
                else:
                    self.dp_world_size = mpu.get_data_parallel_world_size()
            else:
                self.dp_world_size = 1

        # Initialize thread pool instead of single thread
        self.rollout_executor = ThreadPoolExecutor(
            max_workers=self.max_rollout_threads,
            thread_name_prefix="rollout_worker"
        )
        
        # Initialize per-thread task tracking
        with self.thread_task_lock:
            self.thread_task_counts = {i: 0 for i in range(self.max_rollout_threads)}
        
        # Start coordinator thread
        self.coordinator_thread = threading.Thread(
            target=self._coordinator_thread, daemon=True
        )
        self.coordinator_thread.start()

    def destroy(self):
        self.exiting.set()
        
        # Shutdown thread pool
        if self.rollout_executor:
            self.rollout_executor.shutdown(wait=True)
        
        # Wait for coordinator thread
        if self.coordinator_thread:
            self.coordinator_thread.join()

    def get_capacity(self):
        with self.lock:
            max_concurrent_rollouts = max(
                1, self.max_concurrent_rollouts // self.dp_world_size
            )
            
            # Calculate total running tasks across all threads
            with self.thread_task_lock:
                total_running = sum(self.thread_task_counts.values())
            
            capacity = max_concurrent_rollouts - total_running
            
            # Staleness control
            version = self.inference_engine.get_version()
            ofp = self.config.max_head_offpolicyness
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
            consumer_bs = max(1, self.config.consumer_batch_size // self.dp_world_size)
            capacity = min(capacity, (ofp + version + 1) * consumer_bs - sample_cnt)
        return capacity

    def _increment_thread_task_count(self, thread_id: int):
        """Increment task count for a specific thread."""
        with self.thread_task_lock:
            if thread_id not in self.thread_task_counts:
                self.thread_task_counts[thread_id] = 0
            self.thread_task_counts[thread_id] += 1
            
            # Log current thread load distribution
            total_tasks = sum(self.thread_task_counts.values())
            load_info = ", ".join([f"T{tid}:{count}" for tid, count in self.thread_task_counts.items()])
            logger.info(f"🔥 [THREAD_LOAD] Task assigned to Thread-{thread_id}. Current load: [{load_info}] Total: {total_tasks}")

    def _decrement_thread_task_count(self, thread_id: int):
        """Decrement task count for a specific thread."""
        with self.thread_task_lock:
            if thread_id in self.thread_task_counts:
                self.thread_task_counts[thread_id] = max(0, self.thread_task_counts[thread_id] - 1)
                
                # Log updated thread load distribution
                total_tasks = sum(self.thread_task_counts.values())
                load_info = ", ".join([f"T{tid}:{count}" for tid, count in self.thread_task_counts.items()])
                logger.info(f"✅ [THREAD_LOAD] Task completed on Thread-{thread_id}. Current load: [{load_info}] Total: {total_tasks}")

    def _find_available_thread(self) -> int | None:
        """Find thread with the least load for better load balancing."""
        max_tasks_per_thread = max(1, self.max_concurrent_rollouts // self.max_rollout_threads)
        
        with self.thread_task_lock:
            best_thread = None
            min_tasks = float('inf')
            
            # Find the thread with the minimum number of tasks
            for thread_id in range(self.max_rollout_threads):
                current_count = self.thread_task_counts.get(thread_id, 0)
                
                # Only consider threads that haven't hit their max capacity
                if current_count < max_tasks_per_thread and current_count < min_tasks:
                    min_tasks = current_count
                    best_thread = thread_id
            
            return best_thread

    def _rollout_thread(self):
        """Thread that runs the rollout loop."""
        try:
            uvloop.run(self._rollout_thread_async())
        except Exception:
            traceback.print_exc()

    async def _rollout_thread_async(self):
        rollout_tasks = self.rollout_tasks
        rid = 0
        try:
            while not self.exiting.is_set():
                # Check capacity
                capacity = self.get_capacity()
                # Create new rollout task
                self.lock.acquire()
                while (
                    capacity > 0
                    and not self.paused.is_set()
                    and self.input_queue.qsize() > 0
                ):
                    x = self.input_queue.get_nowait()
                    x: _RolloutTaskInput
                    episode_timeout = self.config.episode_timeout_minutes * 60
                    self.logger.debug(f"Get data from puller: {x.data}")
                    task = asyncio.create_task(
                        asyncio.wait_for(
                            x.workflow.arun_episode(self.inference_engine, x.data),
                            timeout=episode_timeout
                        ),
                        name=str(rid),
                    )
                    rollout_tasks[str(rid)] = _RolloutTask(
                        create_time=time.monotonic_ns(), task=task, task_input=x
                    )
                    self.rollout_stat.submitted += 1
                    self.rollout_stat.running += 1
                    if self.config.enable_rollout_tracing:
                        self.logger.info(
                            f"Submit rollout rid {rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                            f"capacity: {capacity}."
                            f"input_queue_size: {self.input_queue.qsize()}."
                        )
                    capacity -= 1
                    rid += 1
                tasks = [x.task for x in rollout_tasks.values()]
                self.lock.release()

                # Wait for rollout completion
                done = []
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                # Collect done results
                for task in done:
                    task_rid = task.get_name()
                    try:
                        traj = await task
                        if isinstance(traj, dict) and all(
                            isinstance(v, CompletionWithTokenLogpReward)
                            for v in traj.values()
                        ):
                            traj = concat_padded_tensors(
                                [v.to_tensor_dict() for v in traj.values()]
                            )
                        assert traj is None or isinstance(traj, TensorDict), traj
                    except asyncio.TimeoutError:
                        logger.warning(f"Episode {task_rid} timed out after {getattr(self.config, 'episode_timeout_minutes', 30)} minutes")
                        # TODO. need to clean up the workflow task to clean up the eks pod

                        traj = None  # Treat timeout as rejected trajectory
                    except Exception as e:
                        logger.error(f"Episode {task_rid} failed with error: {e}")
                        traj = None  # Treat error as rejected trajectory
                    
                    with self.lock:
                        task_obj = rollout_tasks.pop(task_rid)
                        self.rollout_stat.accepted += 1
                        self.rollout_stat.running -= 1
                        if self.config.enable_rollout_tracing:
                            self.logger.info(
                                f"Finish rollout {task_rid}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}."
                                f"capacity: {capacity}."
                                f"input_queue_size: {self.input_queue.qsize()}."
                            )

                    task_input = task_obj.task_input
                    if traj is not None and (
                        task_input.should_accept is None
                        or task_input.should_accept(traj)
                    ):
                        if self.config.enable_rollout_tracing:
                            self.logger.info(
                                f"Accept rollout result of task {task_rid}."
                            )
                        try:
                            self.output_queue.put_nowait(
                                _TimedResult(task_obj.create_time, traj)
                            )
                        except queue.Full:
                            raise RuntimeError(
                                "Output queue full. Please increase queue_size."
                            )
                    else:
                        if self.config.enable_rollout_tracing:
                            self.logger.info(f"Rollout is rejected.")
                        with self.lock:
                            self.rollout_stat.accepted -= 1

                await asyncio.sleep(1)
        except Exception:
            traceback.print_exc()
        finally:
            # Cancel remaining tasks
            with self.lock:
                for task_obj in rollout_tasks.values():
                    if not task_obj.task.done():
                        task_obj.task.cancel()
                        try:
                            await task_obj.task
                        except asyncio.CancelledError:
                            pass

    def _coordinator_thread(self):
        """Coordinator thread that distributes tasks to worker threads."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._coordinator_async())
        except Exception:
            traceback.print_exc()
        finally:
            loop.close()

    async def _coordinator_async(self):
        """Coordinate task distribution and result collection."""
        task_id_counter = 0
        pending_tasks = {}  # task_id -> (task_input, thread_id, future)
        last_summary_time = time.time()
        summary_interval = 10  # Print summary every 10 seconds
        
        while not self.exiting.is_set():
            # Check capacity and assign new tasks
            capacity = self.get_capacity()
            
            while capacity > 0 and not self.paused.is_set() and self.input_queue.qsize() > 0:
                try:
                    task_input = self.input_queue.get_nowait()
                    
                    # Find available thread
                    thread_id = self._find_available_thread()
                    if thread_id is not None:
                        # Increment count BEFORE starting task
                        self._increment_thread_task_count(thread_id)
                        
                        # Submit task to thread pool
                        future = self.rollout_executor.submit(
                            self._execute_episode_in_thread,
                            thread_id,
                            task_input
                        )
                        
                        pending_tasks[task_id_counter] = (task_input, thread_id, future)
                        
                        # Log task assignment details
                        logger.info(f"🚀 [TASK_ASSIGN] Task-{task_id_counter} assigned to Thread-{thread_id}")
                        
                        self.rollout_stat.submitted += 1
                        self.rollout_stat.running += 1
                        
                        if self.config.enable_rollout_tracing:
                            self.logger.info(
                                f"Submit rollout rid {task_id_counter} to thread {thread_id}. "
                                f"Submit: {self.rollout_stat.submitted}, "
                                f"running: {self.rollout_stat.running}, "
                                f"accepted: {self.rollout_stat.accepted}. "
                                f"capacity: {capacity}. "
                                f"input_queue_size: {self.input_queue.qsize()}."
                            )
                        
                        capacity -= 1
                        task_id_counter += 1
                    else:
                        # No available threads, put task back
                        self.input_queue.put(task_input)
                        break
                        
                except queue.Empty:
                    break
            
            # Collect completed tasks
            completed_tasks = []
            for task_id, (task_input, thread_id, future) in list(pending_tasks.items()):
                if future.done():
                    completed_tasks.append((task_id, task_input, thread_id, future))
                    del pending_tasks[task_id]
            
            # Process completed tasks
            for task_id, task_input, thread_id, future in completed_tasks:
                logger.info(f"📥 [TASK_COMPLETE] Task-{task_id} completed on Thread-{thread_id}")
                
                # Decrement count when task completes
                self._decrement_thread_task_count(thread_id)
                self.rollout_stat.running -= 1
                
                try:
                    traj = future.result()
                    
                    if self.config.enable_rollout_tracing:
                        self.logger.info(
                            f"Finish rollout {task_id} from thread {thread_id}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}. "
                            f"input_queue_size: {self.input_queue.qsize()}."
                        )
                    
                    # Process result same as original code
                    if traj is not None and (
                        task_input.should_accept is None
                        or task_input.should_accept(traj)
                    ):
                        if self.config.enable_rollout_tracing:
                            self.logger.info(f"Accept rollout result of task {task_id}.")
                        try:
                            self.output_queue.put_nowait(
                                _TimedResult(time.monotonic_ns(), traj)
                            )
                            self.rollout_stat.accepted += 1
                        except queue.Full:
                            raise RuntimeError("Output queue full. Please increase queue_size.")
                    else:
                        if self.config.enable_rollout_tracing:
                            self.logger.info(f"Rollout {task_id} is rejected.")
                        
                except Exception as e:
                    self.logger.error(f"Task {task_id} from thread {thread_id} failed with error: {e}")
            
            # Print periodic summary
            current_time = time.time()
            if current_time - last_summary_time >= summary_interval:
                with self.thread_task_lock:
                    load_info = ", ".join([f"T{tid}:{count}" for tid, count in self.thread_task_counts.items()])
                    total_tasks = sum(self.thread_task_counts.values())
                logger.info(f"📊 [SUMMARY] Pending tasks: {len(pending_tasks)}, Thread loads: [{load_info}], Total active: {total_tasks}, Queue size: {self.input_queue.qsize()}")
                last_summary_time = current_time
            
            await asyncio.sleep(ROLLOUT_POLL_WAIT_TIME)

    def _execute_episode_in_thread(self, thread_id: int, task_input: _RolloutTaskInput):
        """Execute episode in a worker thread with its own event loop."""
        import threading
        current_thread_name = threading.current_thread().name
        
        try:
            logger.info(f"🎬 [THREAD_START] Thread-{thread_id} ({current_thread_name}) starting episode execution")
            
            # Each thread gets its own event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the episode
            episode_timeout = self.config.episode_timeout_minutes * 60
            
            logger.info(f"⚡ [THREAD_EXEC] Thread-{thread_id} ({current_thread_name}) running arun_episode with timeout {episode_timeout}s")
            
            result = loop.run_until_complete(
                asyncio.wait_for(
                    task_input.workflow.arun_episode(self.inference_engine, task_input.data),
                    timeout=episode_timeout
                )
            )
            
            logger.info(f"🎉 [THREAD_SUCCESS] Thread-{thread_id} ({current_thread_name}) completed episode successfully")
            return result
            
        except asyncio.TimeoutError:
            logger.info(f"⏰ [THREAD_TIMEOUT] Thread-{thread_id} ({current_thread_name}) timed out after {self.config.episode_timeout_minutes} minutes")
            logger.warning(f"Episode in thread {thread_id} timed out after {self.config.episode_timeout_minutes} minutes")
            return None
        except Exception as e:
            logger.info(f"💥 [THREAD_ERROR] Thread-{thread_id} ({current_thread_name}) failed with error: {e}")
            logger.error(f"Episode in thread {thread_id} failed with error: {e}")
            return None
        finally:
            logger.info(f"🏁 [THREAD_END] Thread-{thread_id} ({current_thread_name}) finished execution")
            loop.close()

    def submit(
        self,
        data: Dict[str, Any],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> None:
        try:
            if workflow is None:
                workflow = workflow_builder()
            x = _RolloutTaskInput(
                data=data, workflow=workflow, should_accept=should_accept
            )
            self.input_queue.put_nowait(x)
        except queue.Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(self, count: int, timeout: float | None = None) -> TensorDict:
        tik = time.perf_counter()
        timeout = timeout or float(7 * 24 * 3600)
        while not self.exiting.is_set() and time.perf_counter() - tik < timeout:
            while True:
                # Drain all outputs.
                try:
                    timed_result = self.output_queue.get_nowait()
                    self.result_cache.append(timed_result)
                except queue.Empty:
                    break
            if len(self.result_cache) >= count:
                break
            else:
                time.sleep(ROLLOUT_POLL_WAIT_TIME)
        accepted = len(self.result_cache)
        if self.exiting.is_set():
            raise RuntimeError("Rollout engine is exiting, cannot wait for results.")
        if accepted < count:
            raise TimeoutError(
                f"Timed out waiting for {count} rollouts, only received {accepted}."
            )
        if self.config.enable_rollout_tracing:
            self.logger.info(f"Rollout results are ready!")
        self.result_cache.sort(key=lambda x: x.t)
        results, self.result_cache = (
            self.result_cache[:count],
            self.result_cache[count:],
        )
        random.shuffle(results)
        return concat_padded_tensors([r.data for r in results])

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_builder=workflow_builder,
                should_accept=should_accept,
            )
        return self.wait(count=len(data))

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ):
        if not hasattr(self, "data_generator"):
            self.data_generator = cycle_dataloader(dataloader)
        assert dataloader.batch_size is not None
        while True:
            # Submit at least two batches to allow maximum overlap
            if (
                self.get_capacity() + dataloader.batch_size > 0
                and self.input_queue.qsize() + dataloader.batch_size
                < self.input_queue.maxsize
            ):
                data = next(self.data_generator)
                for item in data:
                    # add a random sleep here. between 1~2 sec
                    import random
                    import time
                    sleep_duration = random.uniform(1.0, 2.0)
                    time.sleep(sleep_duration)
                    self.submit(
                        item,
                        workflow=workflow,
                        workflow_builder=workflow_builder,
                        should_accept=should_accept,
                    )
            try:
                return self.wait(dataloader.batch_size, timeout=1)
            except TimeoutError:
                pass

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()
