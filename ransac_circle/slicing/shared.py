from multiprocessing import shared_memory
import numpy as np

PC_SHM = None
PC_SHAPE = None
PC_DTYPE = None
AXIS_IDX = None
TM_INV = None

def init_worker(shm_name, shape, dtype, axis_idx, tm_inv_flat):
    global PC_SHM, PC_SHAPE, PC_DTYPE, AXIS_IDX, TM_INV

    PC_SHM = shared_memory.SharedMemory(name=shm_name)
    PC_SHAPE = tuple(shape)
    PC_DTYPE = np.dtype(dtype)
    AXIS_IDX = axis_idx
    TM_INV = np.array(tm_inv_flat, dtype=float).reshape(4, 4)