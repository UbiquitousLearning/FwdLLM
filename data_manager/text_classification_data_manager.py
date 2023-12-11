from data_manager.base_data_manager import BaseDataManager
from torch.utils.data import DataLoader
import h5py
import json
import logging
from tqdm import tqdm

class TextClassificationDataManager(BaseDataManager):
    """Data manager for text classification"""
    def __init__(self, args, model_args, preprocessor, process_id=0, num_workers=1):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)

        super(TextClassificationDataManager, self).__init__(args, model_args, process_id, num_workers)
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

        
    def read_instance_from_h5(self, data_file, index_list, desc=""):
        X = list()
        y = list()
        import concurrent.futures
        import threading
        from tqdm import tqdm

        # 定义一个函数来加载单个文件的数据
        def load_data_from_file(idx):
            with data_list_lock:
                X.append(data_file["X"][str(idx)][()].decode("utf-8"))
                y.append(data_file["Y"][str(idx)][()].decode("utf-8"))
                pbar.update(1)
        
        data_list_lock = threading.Lock()  # 用于保护数据列表

        # 创建一个线程池，限制同时活动的线程数量
        # 适当调整max_workers以控制线程数量
        max_workers = 20  # 例如，设置为4以充分利用4个CPU核心
        with tqdm(total=len(index_list)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交文件加载任务
                executor.map(load_data_from_file, index_list)
        return {"X": X, "y": y}
