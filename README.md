# OpenAI Gym BipedalWalker-v2

## 概要  

強化学習の[RL A3C Pytorch Continuous](https://github.com/dgriff777/a3c_continuous/)で、
OpenAI Gym のBipedalWalker-v2を解いたもの。  
このrepositoryの中に２足歩行するBipedalWalkerHardcore-v2の重みファイルが公開されているが、
BipedalWalker-v2は存在しないため、それを求めたもの。  
observationを使わないで高得点を上げている例や１本立ち歩行の例もあるが、
LSTMの有無含め、どのような歩き方になるかを見てみることにしたもの。  
2本足を交互に使って歩くと言う意味では、HardcoreのモデルをBipedalWalker-v2用に再学習したものが、一番良かった。  



## 使い方  

オリジナルの説明文 README_a3c_continous.md を参照のこと。   
自動で終了しないので、画面のLOG出力を見ながら適当なところでctrl-Cキーで強制終了させる。  



オリジナルの設定のBipedalWalker-v2を学習する。　数時間ぐらいかけた。  
```
python main.py --workers 6 --env BipedalWalker-v2 --save-max True --model MLP --stack-frames 1
```

学習した重みファイルを使ってBipedalWalker-v2を動かす。  
```
python gym_eval.py --env BipedalWalker-v2 --num-episodes 100 --stack-frames 1 --model MLP --new-gym-eval True
```

学習した重みファイルを再ロードして、学習を続ける。  
```
python main.py --workers 6 --env BipedalWalker-v2 --load True --save-max True --model MLP --stack-frames 1
```


BipedalWalkerHardcore-v2の重みファイル(CONV1Dモデル, stack-frames 4)を使ってBipedalWalker-v2を動かす。BipedalWalkerHardcore-v2.datをBipedalWalker-v2.datとして上書きしておくこと。  
```
python gym_eval.py --env BipedalWalker-v2 --num-episodes 100 --stack-frames 4 --model CONV --new-gym-eval True
```

BipedalWalkerHardcore-v2の重みファイル(CONV1Dモデル, stack-frames 4)を使ってBipedalWalker-v2を学習する。2-3時間ぐらいかけた。  
。BipedalWalkerHardcore-v2.datをBipedalWalker-v2.datとして上書きしておくこと。  
```
python main.py --workers 6 --env BipedalWalker-v2 --load True --save-max True --model CONV --stack-frames 4
```


１つ前と現在の２つのobservationを使って学習する。  
```
python main.py --workers 6 --env BipedalWalker-v2 --save-max True --model MLP --stack-frames 2
```



## 主な変更点  

- model.pyの中にLSTMのないMLPで学習する設定を追加。  
- test.py 更新した重みファイルを保存する時のメッセージstate_to_saveを追加。  
- shared_optim.py UserWarning: This overload of add_, addcmul_, addcdiv_の対策で引数の順番を変更。  



## 動作環境  

現在のBipedalWalkerのバージョンは３であるが、古いバージョン２を使っている。  
CPUのみ。  


- Ubuntu 18.04 LTS
- python 2.7.17
- torch==1.5.0+cpu
- torchvision==0.5.0+cpu
- torchaudio==0.4.0
- numpy==1.16.6
- gym==0.10.11
- Box2D==2.3.2
- pyglet==1.3.2
- pyyaml==3.12
- setproctitle==1.1.10
- typing==3.7.4.3



また、Google colab上で実行するスクリプトを作ってみた。  
[BipedalWalker_v2_Colab.ipynb](https://colab.research.google.com/github/shun60s/BipedalWalker-a3c_continuous-clone/blob/master/BipedalWalker_v2_Colab.ipynb)  


### trained_models   

BipedalWalker-v2.dat　オリジナルの設定で学習した重みファイル  
BipedalWalker-v2_withoutLSTM.dat　LSTMのないMLPで学習した重みファイル  
BipedalWalker-v2_stackframe2.dat　stack_frame=2で学習した重みファイル  
BipedalWalkerHardcore-v2.dat　オリジナルからcloneした重みファイル  
BipedalWalker-v2_trained_using_Hardcore_dat.dat 上記のをHardcoreのモデルを、更にBipedalWalker環境で学習させたもの。これが2本足を交互に使って走るという意味では、これが一番、良かった。  
![BipedalWalker-v2_trained_using_Hardcore_dat  mp4 sample](https://user-images.githubusercontent.com/36104188/104119919-48597280-5376-11eb-8ed5-e77576a5ad12.mp4)  
2本足を使って歩くには、環境（Hardcoreの落とし穴のような1本歩行では不可能な環境）とそれなりのDNNの構成(MLPだけでは2本足で交互に歩くは不可なのか？)が揃わないといけないのかもしれない。  

BipedalWalker-v2_monitor_xxxの中に　歩き方の画像をmp4で格納した。  


## ライセンス  
Apache License 2.0  
オリジナルのライセンス文 LICENSE_ac3_continous.MD を参照のこと。   

