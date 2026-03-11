参考资料：
[1] I. G. Cumming and F. H. Wong, Digital Processing of Synthetic Aperture Radar Data: Algorithms and Implementation. Norwood, MA, USA: Artech House, 2005.
[2] Ian G. Cumming, Frank H. Wong 著. 洪文, 胡东辉, 等译. 《合成孔径雷达成像——算法与实现》[M]. 北京: 电子工业出版社, 2019.
[3]https://github.com/Hanyang0603/Synthetic-Aperture-Radar-Imaging-Algorithms
[4]https://github.com/HAMGZZ/SAR-RDA

更新日期:2026/03/10
作者:Ashorein6

SAR_RDA_Algorithm
实现步骤：
1.将雷达原始数据进行解调,解调后的点目标信号模型为教材式(6.1),方位包络为sinc平方函数
2.距离公式使用双曲线方程(式6.1)而不是抛物线方程
3.距离向FFT,并进行标准的距离匹配滤波(式3.35),不进行傅里叶逆变换
4.方位向FFT变换至二维频域,进行二次距离压缩Src(式6.27)
5.距离向IFFT变换至距离多普勒域
6.距离徙动校正RCRC(式6.25)
7.方位向匹配滤波(式6.26)
8.方位向IFFT
9.计算PSLR,IRW,ISLR进行点目标分析