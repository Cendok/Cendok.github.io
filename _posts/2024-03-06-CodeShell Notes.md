---
layout: post
title: Codeshell Notes
description: CodeShell下载-部署-使用教程
categories: Notes
tags: [Notes]
---

# CodeShell下载-部署-使用教程

[codeShell本地电脑搭建模型及实验vscode插件 - 简书 (jianshu.com)](https://www.jianshu.com/p/887e65d11d6b)



## 下载

[codeShell_免费高速下载|百度网盘-分享无限制 (baidu.com)](https://pan.baidu.com/s/1BcYpN5zAZlshcnGxg-QXnw?pwd=cvyc#list/path=%2F)

```
链接：https://pan.baidu.com/s/1BcYpN5zAZlshcnGxg-QXnw?pwd=cvyc 
提取码：cvyc 
```



## 部署

### VScode上添加插件

#### 方法1

VScode上直接安装插件



#### 方法2编译vscode插件

或者通过以下步骤进行添加插件到VScode

![image-20240306215901923](/images/posts/2024-03-06-CodeShell/image-20240306215901923.png)



生成插件文件,并直接把codeshell-vscode-0.0.2.vsix拖动到vscode插件列表中，重启vscode，就可以看到插件了



### 通过w64devkit编译模型服务

把这个文档**codeshell-chat-q4_0.gguf**拖进以下路径中，再运行命令即可：

![image-20240306224141997](/images/posts/2024-03-06-CodeShell/image-20240306224141997.png)

![image-20240306224113782](/images/posts/2024-03-06-CodeShell/image-20240306224113782.png)



```
@ext:WisdomShell.codeshell-vscode
```

![image-20240306222821655](/images/posts/2024-03-06-CodeShell/image-20240306222821655.png)



![image-20240308141002390](/images/posts/2024-03-06-CodeShell/image-20240308141002390.png)





## 使用

### 打开w64devkit.exe文件

```
路径：
E:\CodeShell\w64devkit
```

需要一直通过w64devkit输入以下命令连接远程服务器，才能使用实时对话功能。



运行w64devkit.exe后，进入llama_cpp_for_codeshell目录



### 在对应路径下输入命令

```
切换倒E盘，直接cd命令没法切换：
C:\Users\>E:
E:\>

路径：
E:\CodeShell\llama_cpp_for_codeshell-master\models
#2026.3.11测试以下路径可行
E:\CodeShell\llama_cpp_for_codeshell-master

启动服务命令：
./server -m ./models/codeshell-chat-q4_0.gguf --host 127.0.0.1 --port 8080

默写错误：
./server -m /codeshell/.gglf --host 127.0.0.1 --port 8080
```



显示这个状态，远程服务器的端口`8080`就成功连接本地计算机的端口

![image-20240309211910721](/images/posts/2024-03-06-CodeShell/image-20240309211910721.png)



否则显示这个界面

![image-20240309212029220](/images/posts/2024-03-06-CodeShell/image-20240309212029220.png)



## 额外发现llama.cpp - chat

[llama.cpp - chat](http://127.0.0.1:8080/)

点击这个连接可以进入可视化对话界面

![image-20240309212200626](/images/posts/2024-03-06-CodeShell/image-20240309212200626.png)



## 可能遇到的问题：

VScode使用Codeshell时报错——因为没有连接到服务器端口

```
f: request to http://127.0.0.1:8080/completion failed, reason: connect ECONNREFUSED 127.0.0.1:8080
```



### 解决

通过w64devkit启动模型服务，运行w64devkit.exe后，进入llama_cpp_for_codeshell目录，输入

```
E:/CodeShell/llama_cpp_for_codeshell-master $ ./server -m E:/CodeShell/llama_cpp_for_codeshell-master/models/codeshell-chat-q4_0.gguf --http://127.0.0.1 --port 8080

报错：sh: ./server: not found
```


