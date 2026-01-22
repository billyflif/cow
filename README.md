在jetson上运行的时候，因为没有现成的轮子，所以奥比中光和torch的轮子需要自己手动编译，由于坑比较多，所以建议复用我这已经编译好的，其中，奥比中光的SDK轮子在git项目中已经上传，jetson的torch和torchvision轮子文件太大，没有加入。可以在如下链接中下载：
通过网盘分享的文件：torch(jetson)
链接: https://pan.baidu.com/s/1lAxuk7YmjhFXm25-Mq_gng?pwd=1234 提取码: 1234。此外，还有已经容器化的Docker文件，下载地址： https://pan.baidu.com/s/1717l-2fzNT0VPig1vkogEg?pwd=1234 提取码: 1234。要注意：原本的识别系统代码中的路径是绝对路径，如果要在容器中运行，需要修改容器中的路径,例如：sudo docker run -it --rm --runtime nvidia --network host --ipc host --privileged \
  -v /home/nvidia/PycharmProjects/Obc_SDK:/home/nvidia/PycharmProjects/Obc_SDK \
  cowreid:jp461-py38。或者，修改代码，使得所有路径都用相对路径。
  项目的权重要注意：使用numpy2.x训练的话，需要使用convert_reid_checkpoint.py进行转化，因为jetson上只支持numpy1.x,权重由于更新快，不通用，所以不再挂出单独下载链接。