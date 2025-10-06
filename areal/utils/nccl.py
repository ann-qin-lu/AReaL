import datetime
import os 
# For large models, generation may consume more than 3600s.
# We set a large value to avoid NCCL timeout issues during generaiton.
timeout=int(os.getenv("NCCL_DEFAULT_TIMEOUT", 2100))
# reduce to 35 mins to prevent stuck job detection in batch
NCCL_DEFAULT_TIMEOUT = datetime.timedelta(seconds=timeout)
