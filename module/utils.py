from typing import Optional
import torch
from subprocess import check_output

def flush_gpu(
        text : Optional[str] = None,
        ) -> None:
        """
        Frees memory on the GPU by deleting some variables and freeing the cache.
        """       
        torch.cuda.empty_cache() 
        torch.cuda.ipc_collect()
        if text is not None:
            result = check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'], encoding='utf-8')
            memory_footprint = [int(x) for x in result.strip().split('\n')][0]
            print(text, memory_footprint)