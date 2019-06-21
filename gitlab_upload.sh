#!/usr/bin/env bash
#cd ./Awesome_Computer_Vision/
#sudo rm -r .git
#git init
git remote rm origin
git config --global user.name "零捌"
git config --global user.email "taihong.cth@autonavi.com"

git remote add origin git@gitlab.alibaba-inc.com:amap-xlab/poi-grouping.git
#git remote add origin git@gitlab.alibaba-inc.com:taihong.cth/POIReconstruction.git

git add  *.py
git commit -m "20190423更新：商汤提出Switchable Whitening，相比BN GN LN等均有良好表现"
git push -u origin master -f

echo "over!"