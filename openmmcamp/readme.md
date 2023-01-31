本文件夹存放 OpenMMLab camp 的相关代码。

# 相关资料 & 注意事项
作业提交：在对应 repo 的 issue 地址回复完成的作业代码仓库。
- [寒假课程前期资料](https://aicarrier.feishu.cn/docs/doccnP7NPMfRr9TAcwRsPKgkOgc)
    
    记录一下这个文档里比较有价值的内容：
    - 深入浅出PyTorch [doc](https://datawhalechina.github.io/thorough-pytorch/) | [video](https://www.bilibili.com/video/BV1L44y1472Z/)
    - 如何提交规范代码 [video](https://www.bilibili.com/video/BV1JD4y1v77A/) | [code](https://github.com/HAOCHENYE/yehc-tutorial/)
        - [pull-request-vs-merge-request](https://stackoverflow.com/questions/22199432/pull-request-vs-merge-request)：等效
        - [github 操作](https://aicarrier.feishu.cn/file/boxcn53wyyHg2Jv0iCOwMQJ7j9b) 即 `./tutorials/熟悉GitHub.pdf`
    - OpenMMLab 基础库 MMCV [贡献代码](https://mmcv.readthedocs.io/zh_CN/latest/community/contributing.html) | [代码规范](https://mmcv.readthedocs.io/zh_CN/latest/community/code_style.html) | [介绍与安装](https://mmcv.readthedocs.io/zh_CN/latest/get_started/article.html)
- [OpenMMLabCourse](https://github.com/open-mmlab/OpenMMLabCourse)

接下来可能会用到的git命令：
```bash
# 配置信息
git config --global user.name "name"
git config --global user.email "email"

git fetch origin # 或者：git pull origin，但前者会存放在本地的 origin/main 后者会合并到本地的 main
git checkout main # 切换到 main 分支（默认是 main 分支）
git checkout -b dev # 新建分支

# .... do some changes ....
git add .
git commit -m "message"

git checkout main # 切回原分支/主干分支
git merge dev # 合并修改

# 如果发生冲突，手工处理冲突然后执行：
git merge --continue    # 解决完冲突后继续合并
git merge --abort   # 放弃本次合并，回滚到合并前状态


git remote -v # 查看关联的远程仓库信息
```
