import argparse
import os
import glob
import datetime
import random
import shutil

class SplitTrainData:
    def __init__(self, args):
        self.dir_path = args.dir_path
        self.save_path = args.save_path
        
        # 分割割合
        self.rates = {
            "train":args.trainrate,
            "val":args.valrate,
            "test":args.testrate
        }
        
        # os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path,"train"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path,"test"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path,"val"), exist_ok=True)
        
        self.ext = args.ext
    
    def getImagePathList(self):
        files = glob.glob(f"{self.dir_path}/*.{self.ext}")
        self.basenames = [os.path.splitext(os.path.basename(path))[0] for path in files]
        random.shuffle(self.basenames)
        # print(self.files[0:10])
    
    def run_split(self):
        filenum = len(self.basenames)
        
        train_num = round(filenum * self.rates["train"])
        val_num = round(filenum * self.rates["val"])
        test_num = round(filenum * self.rates["test"])
        
        # print(train_num, val_num, test_num)
        
        datas = {
            "train":self.basenames[0:train_num],
            "val"  :self.basenames[train_num:train_num+val_num],
            "test" :self.basenames[train_num+val_num:]
        }
        for key, val in datas.items():
            for i, basename in enumerate(val):
                # 画像のコピー
                shutil.copy(f"{self.dir_path}/{basename}.{self.ext}", f"{self.save_path}/{key}/{key}_{i}.{self.ext}")
                # アノテーションデータのコピー
                if os.path.isfile(f"{self.dir_path}/{basename}.txt"):
                    shutil.copy(f"{self.dir_path}/{basename}.txt", f"{self.save_path}/{key}/{key}_{i}.txt")
                # print(f"({data} => {self.save_path}/{key}/{key}_{i}.{self.ext}")
            shutil.copy(f"{self.dir_path}/classes.txt", f"{self.save_path}/{key}/classes.txt")
        
        with open(f"{self.dir_path}/classes.txt") as f:
            l = [x.rstrip("\n") for x in f.readlines()]
            # print(l)
        
        with open(f"{self.save_path}/model.yaml", mode="w") as f:
            f.write("train: ./train\n")
            f.write("val: ./val\n")
            f.write("test: ./test\n")
            f.write(f"nc: {len(l)}\n")
            f.write(f"names: {l}\n")
            

if __name__=="__main__":
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dir_path")
    parser.add_argument("--save_path", 
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            f"results/{now.strftime('%y%m%d_%H%M%S')}")
                        )
    
    parser.add_argument("-train", "--trainrate", default=0.7)
    parser.add_argument("-val", "--valrate", default=0.1)
    parser.add_argument("-test", "--testrate", default=0.2)
    parser.add_argument("--ext", default="jpg")
    
    
    
    args = parser.parse_args()
    
    splits = SplitTrainData(args)
    
    splits.getImagePathList()
    splits.run_split()