import argparse
import os
import glob
import datetime
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

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
        
        with open(f"{self.dir_path}/classes.txt") as f:
            # classlist = f.readlines()
            self.classlist = np.array(f.read().splitlines())

        
        print(f"Number of Class : {len(self.classlist)}")
    
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
    
        self.run_write_yaml(l)
        
    def run_write_yaml(self, l):
        with open(f"{self.save_path}/data.yaml", mode="w") as f:
            f.write("# Directory path of Images\n")
            f.write("train: ./train\n")
            f.write("val: ./val\n")
            f.write("test: ./test\n")
            
            f.write("# Number of class\n")
            f.write(f"nc: {len(l)}\n")
            
            f.write("# Class names\n")
            f.write(f"names: {l}\n")
    
    def count_class(self):
        countlist = np.zeros(len(self.classlist),dtype=np.int64)

        files = glob.glob(f"{self.dir_path}/*.txt")
        
        for i, path in enumerate(files):
            if os.path.basename(path) == "classes.txt":
                continue
            with open(path) as f:
                lines = f.readlines()
                # print(lines)
                for line in lines:
                    line = line.split()
                    countlist[int(line[0])] += 1
        # print(countlist)
        
        np_classlist = np.array([],dtype=np.int64)
        np_countlist = np.array([],dtype=np.int64)
        for i in range(0,len(countlist)):
            if countlist[i] == 0:
                continue
            np_classlist = np.append(np_classlist,self.classlist[i])
            np_countlist = np.append(np_countlist,countlist[i])
        
        fig, ax = plt.subplots(1,1)
        
        # ランダムなRGBのリストを作成
        colors = [[np.random.rand(), np.random.rand(), np.random.rand()] for i in np_classlist]
        
        bars = ax.barh(np_classlist, np_countlist, align="center", color=colors)
        
        # 各棒グラフの上に数値を表示
        for bar, count in zip(bars, np_countlist):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{count}', va='center', ha='left', fontsize=10, color='black')
        
        ax.set_title("Count of Class")
        ax.invert_yaxis()
        
        # グラフの最大値のサイズを調整
        ax.set_xlim(0, max(np_countlist) * 1.1)
        
        fig.tight_layout()
        fig.savefig(f"{self.save_path}/class.png")


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
    splits.count_class()