---
layout: post
title: Face-Comparison-System-V2
description: 基于CNN网络和Flask框架的手写数字识别系统
categories: System
tags: [System]
---

> Face-Comparison-System-V2



附上码源：

[Cendok/Face-Comparison-System-V2](https://github.com/Cendok/Face-Comparison-System-V2)



# README-Face-Comparison-System-V2



## Pycharm中新建虚拟环境

Python3.9(Face-Comparison-System)

V2共用同一个虚拟环境

```
pip install Flask==2.2.1 opencv-python==4.6.0.66 torch==1.12.1 facenet-pytorch==2.5.2

pip install Werkzeug==2.0.3
pip install Flask==1.1.2 Werkzeug==1.0.1
pip install Jinja2==2.11.3
pip install MarkupSafe==2.0.1
pip install itsdangerous==1.1.0
pip install numpy==1.21.0
```



| 软件包             | 版本      |
| ------------------ | --------- |
| Flask              | 1.1.2     |
| Jinja2             | 2.11.3    |
| MarkupSafe         | 2.0.1     |
| Werkzeug           | 1.0.1     |
| certifi            | 2026.2.25 |
| charset-normalizer | 3.4.6     |
| click              | 8.1.8     |
| colorama           | 0.4.6     |
| facenet-pytorch    | 2.5.2     |
| idna               | 3.11      |
| importlib-metadata | 8.7.1     |
| itsdangerous       | 1.1.0     |
| numpy              | 1.21.0    |
| opencv-python      | 4.6.0.66  |
| pillow             | 11.3.0    |
| pip                | 23.2.1    |
| requests           | 2.32.5    |
| setuptools         | 68.2.0    |
| torch              | 1.12.1    |
| torchvision        | 0.13.1    |
| typing-extensions  | 4.15.0    |
| urllib3            | 2.6.3     |
| wheel              | 0.41.2    |
| zipp               | 3.23.0    |



## app.py

沿用Face-Comparison-System



改端口1235

```python
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=1235)
```



```python
#设置成能找到静态文件的路径，CSS、JavaScript、图像等
app = Flask(__name__, static_url_path="/static")
```



### 问题-显示二进制文件，报404

从你发来的日志和网页截图来看，大部分资源（如CSS和JavaScript文件，以及一些图片）无法被找到，导致返回404错误。

```bash
Running on http://0.0.0.0:1235/ (Press CTRL+C to quit)
127.0.0.1 - - [21/Mar/2026 19:57:33] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [21/Mar/2026 19:57:33] "GET /static/css/bootstrap.min.css HTTP/1.1" 404 -
127.0.0.1 - - [21/Mar/2026 19:57:33] "GET /static/css/fontawesome-all.min.css HTTP/1.1" 404 -
127.0.0.1 - - [21/Mar/2026 19:57:33] "GET /static/css/animate.css HTTP/1.1" 404 -
127.0.0.1 - - [21/Mar/2026 19:57:33] "GET /static/css/slick.css HTTP/1.1" 404 -
```

![显示二进制网页](E:\starta\Face-Comparison-System-V2\显示二进制网页.png)



检查Web框架设置：使用的是Flask，可以检查如下一行

```python
app = Flask(__name__, static_url_path="/")

app = Flask(__name__, static_url_path="/static")

#设置成能找到静态文件的路径，CSS、JavaScript、图像等
```

修改路径





## index.html

改名称

```html
<head>
<title>Face-Comparison-System-V2</title>
</head>
```



修剪枝叶，保留想要的功能



对比功能放在<body></body>内

```html
<section class="bg-upcoming-events">
            <div class="container">
                <div class="row">
                    <div class="upcoming-events">
                        <div class="section-header">
                            <h1>&#128519;</h1>
                            <p>上传待判断是否为同一人的两张脸部图像，点击预测按钮进行比对分析</p>
                        </div>
                        <style>
                            .section-header {
                                text-align: center; /* 使文本居中 */
                                margin: 20px; /* 添加一些外边距，便于视觉效果 */
                            }
                        </style>
                        <!-- .section-header -->
                        <div class="row">
                            <div class="col-lg-6">
                                <h3 style="color: black;">待比对图片左</h3>
                                <div>
                                    <!--                 href="javascript:;"-->
                                    <input href="javascript:;" class="btn btn-default" tabindex="0" type="file" name="file"
                                           id="file0">

                                    </input>
                                    <p></p>
                                    <img src="" id="img0">
                                </div>
                            </div>
                            <!-- .col-lg-6 -->
                            <div class="col-lg-6">
                                <h3 style="color: black;">待比对图片右</h3>
                                <div>
                                    <!--                 href="javascript:;"-->
                                    <input href="javascript:;" class="btn btn-default" tabindex="0" type="file" name="file"
                                           id="file1">
                                    </input>
                                    <p></p>
                                    <img src="" id="img1">
                                </div>
                            </div>

                            <!-- .col-lg-6 -->
                            <p></p>
                            <div>
                                <!--                style="margin-top:20px;width: 35rem;height: 30rem; padding-left: 20px"-->
                                <input class="btn btn-default" type="button" id="b0"
                                       onclick="test0()" style="color: #000000"
                                       value="点击预测">
                                <p></p>
                                <pre id="out">点击预测获取结果</pre>
                                <!--                <pre id="out" style="width:320px;height:50px;line-height: 50px;margin-top:20px;"></pre>-->
                            </div>
                        </div>
                        <!-- .row -->
                    </div>
                    <!-- .upcoming-events -->
                </div>
                <!-- .row -->
            </div>
            <!-- .container -->
        </section>
```



Javascript文件就用模板自带的

```html
<!-- All js here -->
<script src="../static/js/modernizr-3.5.0.min.js"></script>
<script src="../static/js/jquery-1.12.4.min.js"></script>
<script src="../static/js/popper.min.js"></script>
<script src="../static/js/bootstrap.min.js"></script>
<script src="../static/js/one-page-nav-min.js"></script>
<script src="../static/js/slick.min.js"></script>
<script src="../static/js/wow.min.js"></script>
<script src="../static/js/plugins.js"></script>
<script src="../static/js/jquery.meanmenu.min.js"></script>
<script src="../static/js/main.js"></script>
```





功能性的自己写，自己加，放在<body></body>内

```html
        <script>
    /*https://unpkg.com/live2d-widget-model-shizuku@1.0.5/assets/shizuku.model.json*/
    L2Dwidget.init({ "model": { jsonPath:
          "https://unpkg.com/live2d-widget-model-koharu@1.0.5/assets/koharu.model.json",
        "scale": 1 }, "display": { "position": "left", "width": 110, "height": 150,
        "hOffset": 0, "vOffset": -20 }, "mobile": { "show": true, "scale": 0.5 },
      "react": { "opacityDefault": 0.8, "opacityOnHover": 0.1 } });
        </script>

        <script type="text/javascript">
        $("#file0").change(function () {
            var objUrl = getObjectURL(this.files[0]);//获取文件信息
            console.log("objUrl = " + objUrl);
            if (objUrl) {
                $("#img0").attr("src", objUrl);
            }
        });
        $("#file1").change(function () {
            var objUrl = getObjectURL(this.files[0]);//获取文件信息
            console.log("objUrl = " + objUrl);
            if (objUrl) {
                $("#img1").attr("src", objUrl);
            }
        });

        function test0() {
            var fileobj0 = $("#file0")[0].files[0];
            var fileobj1 = $("#file1")[0].files[0];
            var form = new FormData();
            form.append("file0", fileobj0);
            form.append("file1", fileobj1);
            var out = '';
            var fazhi = '';
            var oushi = '';
            $.ajax({
                type: 'POST',
                url: "predict",
                data: form,
                async: false,       //同步执行
                processData: false, // 告诉jquery要传输data对象
                contentType: false, //告诉jquery不需要增加请求头对于contentType的设置
                success: function (arg) {
                    console.log(arg)
                    out = arg.fazhi + '\n' + arg.oushi + '\n' + arg.result
                    err = arg.err
                }, error: function () {
                    console.log("后台处理错误");
                }
            });

            // if(oushi!==undefined){document.getElementById("oushi").innerText = oushi;}
            // if(fazhi!==undefined){document.getElementById("fazhi").innerText = fazhi;}
            if (err !== undefined) {
                document.getElementById("out").innerText = err;
            }
            if (out !== undefined) {
                document.getElementById("out").innerHTML = out;
            }

        }

        function getObjectURL(file) {
            var url = null;
            if (window.createObjectURL != undefined) {
                url = window.createObjectURL(file);
            } else if (window.URL != undefined) { // mozilla(firefox)
                url = window.URL.createObjectURL(file);
            } else if (window.webkitURL != undefined) { // webkit or chrome
                url = window.webkitURL.createObjectURL(file);
            }
            return url;
        }
        </script>
```



## 访问

```
http://localhost:1235/

默认
http://127.0.0.1:1235/
```



## 界面展示

![display1](E:\starta\Face-Comparison-System-V2\display1.png)

![display2](E:\starta\Face-Comparison-System-V2\display2.png)


