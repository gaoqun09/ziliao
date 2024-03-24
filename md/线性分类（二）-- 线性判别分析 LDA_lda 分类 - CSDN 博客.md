> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/xq151750111/article/details/121341117?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166787537216782428646676%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166787537216782428646676&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-121341117-null-null.142%5Ev63%5Ewechat,201%5Ev3%5Econtrol,213%5Ev1%5Et3_esquery_v1&utm_term=LDA&spm=1018.2226.3001.4187)

>         在机器学习领域，LDA 是两个常用模型的简称：[线性](https://so.csdn.net/so/search?q=%E7%BA%BF%E6%80%A7&spm=1001.2101.3001.7020)判别分析（Linear Discriminant Analysis） 和隐含狄利克雷分布（Latent Dirichlet Allocation）。在自然语言处理领域，隐含狄利克雷分布是一种处理文档的主题模型。本文只讨论线性判别分析，在模式识别领域（比如人脸识别，舰艇识别等图形图像识别领域）中有非常广泛的应用。

        线性判别分析（Linear Discriminant Analysis，简称 LDA）是一种经典的监督学习的数据[降维](https://so.csdn.net/so/search?q=%E9%99%8D%E7%BB%B4&spm=1001.2101.3001.7020)方法，也叫做 Fisher 线性判别（Fisher Linear Discriminant，FLD），是模式识别的经典算法，它是在 1996 年由 Belhumeur 引入模式识别和人工智能领域的。LDA 的主要思想是**将一个高维空间中的数据投影到一个较低维的空间中，且投影后要保证各个类别的类内方差小而类间均值差别大，**这意味着同一类的高维数据投影到低维空间后相同类别的聚在一起，而不同类别之间相距较远。如下图将二维数据投影到一维直线上：

![](https://www.biaodianfu.com/wp-content/uploads/2020/09/lda.png??x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)

  
        LDA 降维的目标：将带有标签的数据降维，投影到低维空间同时满足三个条件：

1.  尽可能多的保留数据样本的信息（即选择最大的特征是对应的特征向量所代表的方向）。
2.  寻找使样本尽可能好分的最佳投影方向。
3.  投影后使得同类样本尽可能近，不同类样本尽可能远。

#### 1 瑞利商（Rayleigh quotient）与广义瑞利商（genralized Rayleigh quotient）

        首先来看看瑞利商的定义。瑞利商是指这样的函数 R ( A , x ) R(\pmb{A},\pmb{x}) R(AAA,xxx)：  
R ( A , x ) = x H A x x H x (1-1) R(\boldsymbol{A}, \boldsymbol{x})=\dfrac{\boldsymbol{x}^{H} \boldsymbol{A x}}{\boldsymbol{x}^{H} \boldsymbol{x}} \tag{1-1} R(A,x)=xHxxHAx​(1-1)

        其中 x \pmb{x} xxx 为非零向量，而 A \pmb{A} AAA 为 n × n n\times n n×n 的`Hermitan`矩阵。所谓的`Hermitan`矩阵就是满足共轭转置矩阵和自己相等的矩阵，即 A H = A \pmb{A}^H=\pmb{A} AAAH=AAA。如果我们的矩阵 A \pmb{A} AAA 是实矩阵，则满足 A T = A \pmb{A}^T =\pmb{A} AAAT=AAA 的矩阵即为`Hermitan`矩阵。

        瑞利商 R ( A , x ) R(\pmb{A},\pmb{x}) R(AAA,xxx) 有一个非常重要的性质，即它的最大值等于矩阵 A \pmb{A} AAA 最大的特征值，而最小值等于矩阵 A \pmb{A} AAA 的最小的特征值，也就是满足  
λ min ⁡ ≤ x H A x x H x ≤ λ max ⁡ (1-2) \lambda_{\min } \leq \dfrac{\boldsymbol{x}^{H} \boldsymbol{A x}}{\boldsymbol{x}^{H} \boldsymbol{x}} \leq \lambda_{\max }\tag{1-2} λmin​≤xHxxHAx​≤λmax​(1-2)

        具体的证明可以参考：[瑞利商（Rayleigh Quotient）及瑞利定理（Rayleigh-Ritz theorem）的证明](https://blog.csdn.net/klcola/article/details/104800804)

        当向量 x \pmb{x} xxx 是标准正交基时，即满足 x H x = 1 \pmb{x}^H\pmb{x}=1 xxxHxxx=1 时，瑞利商退化为： R ( A , x ) = x H A x R(\pmb{A},\pmb{x})=\pmb{x}^H\pmb{Ax} R(AAA,xxx)=xxxHAxAxAx。若 x \pmb{x} xxx 是实向量，则可以构造构造拉格朗日乘子函数 L ( x , λ ) = x T A x + λ ( x T x − 1 ) L(\boldsymbol{x},\lambda)=\boldsymbol{x}^T\boldsymbol{Ax}+\lambda(\boldsymbol{x}^T\boldsymbol{x}-1) L(x,λ)=xTAx+λ(xTx−1)

        对 x \pmb{x} xxx 求梯度并令梯度为 0 可以得到 2 A x + 2 λ x = 0 2\pmb{Ax}+2\lambda \pmb{x}=0 2AxAxAx+2λxxx=0，等价于 A x = λ ^ x \pmb{Ax}=\hat{\lambda} \pmb{x} AxAxAx=λ^xxx，假设 λ ^ i \hat{\lambda}_i λ^i​ 是 A \pmb{A} AAA 的第 i i i 个特征值， x i \pmb{x}_i xxxi​ 是其对应的特征向量。

        带入瑞利商的定义，可得 R ( A , x i ) = λ ^ i R(\pmb{A},\pmb{x}_i)=\hat{\lambda}_i R(AAA,xxxi​)=λ^i​，在最大的特征值处，瑞利商有最大值，在最小的特征值处，瑞利商有最小值。上面的形式在谱聚类和 PCA 中都有出现。

        下面来看广义瑞利商。广义瑞利商是指这样的函数 R ( A , B , x ) R(\pmb{A},\pmb{B},\pmb{x}) R(AAA,BBB,xxx)：  
R ( A , B , x ) = x H A x x H B x (1-3) R(\pmb{A},\pmb{B}, \pmb{x})=\frac{\pmb{x}^{H} \pmb{A x}}{\pmb{x}^{H} \pmb{B x}} \tag{1-3} R(AAA,BBB,xxx)=xxxHBxBxBxxxxHAxAxAx​(1-3)

        其中 x \pmb{x} xxx 为非零向量，而 A , B \pmb{A},\pmb{B} AAA,BBB 为 n × n n\times n n×n 的`Hermitan`矩阵。 B \pmb{B} BBB 为正定矩阵。它的最大值和最小值是什么呢？其实我们只要通过将其通过标准化就可以转化为瑞利商的格式。  
        假设对任意非 0 向量，有 x T B x > 0 \pmb{x}^T\pmb{Bx}>0 xxxTBxBxBx>0，如果令 B = C C T \pmb{B}=\pmb{CC}^T BBB=CCCCCCT ，这是对矩阵 B \pmb{B} BBB 的 [Cholesky 分解](https://blog.csdn.net/acdreamers/article/details/44656847)，同时令 x = ( C T ) − 1 y \pmb{x}=(\pmb{C}^T)^{-1}\pmb{y} xxx=(CCCT)−1y​y​​y，则可以将广义瑞利商转化成瑞利商的形式  
x T A x x T B x = ( ( C T ) − 1 y ) T A ( ( C T ) − 1 y ) ( ( C T ) − 1 y ) T C C T ( ( C T ) − 1 y ) = y T C − 1 A ( C T ) − 1 y y T y (1-4) \frac{\pmb{x}^T\pmb{Ax}}{\pmb{x}^T\pmb{Bx}}=\frac{((\pmb{C}^T)^{-1}\pmb{y})^T\pmb{A}((\pmb{C}^T)^{-1}\pmb{y})}{((\pmb{C}^T)^{-1}\pmb{y})^T\pmb{CC}^T((\pmb{C}^T)^{-1}\pmb{y})}=\frac{\pmb{y}^T\pmb{C}^{-1}\pmb{A}(\pmb{C}^T)^{-1}\pmb{y}}{\pmb{y}^T\pmb{y}} \tag{1-4} xxxTBxBxBxxxxTAxAxAx​=((CCCT)−1y​y​​y)TCCCCCCT((CCCT)−1y​y​​y)((CCCT)−1y​y​​y)TAAA((CCCT)−1y​y​​y)​=y​y​​yTy​y​​yy​y​​yTCCC−1AAA(CCCT)−1y​y​​y​(1-4)

        用前面的瑞利商的性质，我们可以很快的知道， R ( A , B , x ) R(\pmb{A},\pmb{B},\pmb{x}) R(AAA,BBB,xxx) 的最大和最小值由矩阵 C − 1 A ( C T ) − 1 \pmb{C}^{-1}\pmb{A}(\pmb{C}^T)^{-1} CCC−1AAA(CCCT)−1 的最大和最小特征值决定。

        加上等式约束 x T B x = 1 {\pmb{x}^T\pmb{Bx}}=1 xxxTBxBxBx=1，广义瑞利商为 R ( A , B , x ) = x T A x R(\pmb{A},\pmb{B},\pmb{x})={\pmb{x}^T\pmb{Ax}} R(AAA,BBB,xxx)=xxxTAxAxAx，构造拉格朗日乘子函数 L ( x , λ ) = x T A x + λ ( x T B x − 1 ) L(\pmb{x},\lambda)={\pmb{x}^T\pmb{Ax}}+\lambda({\pmb{x}^T\pmb{Bx}}-1) L(xxx,λ)=xxxTAxAxAx+λ(xxxTBxBxBx−1)

        对 x \pmb{x} xxx 求梯度并令梯度为 0 可以得到： 2 A x + 2 λ B x = 0 2\pmb{Ax}+2\lambda\pmb{Bx}=0 2AxAxAx+2λBxBxBx=0，这等价于 A x = λ ^ B x \pmb{Ax} = \hat{\lambda}\pmb{Bx} AxAxAx=λ^BxBxBx，这是广义特征值的问题，如果 B \pmb{B} BBB 可逆，可得 B − 1 A x = λ ^ x \pmb{B}^{-1}\pmb{Ax}=\hat{\lambda} \pmb{x} BBB−1AxAxAx=λ^xxx，因此广义瑞利商的所有极值在广义特征值处取得，假设 λ ^ i \hat{\lambda}_i λ^i​ 是第 i i i 个广义特征值， x i \pmb{x}_i xxxi​ 是其对应的广义特征向量，带入广义瑞利商的定义可得 R ( A , B , x i ) = λ i R(\pmb{A},\pmb{B},\pmb{x}_i)=\lambda_i R(AAA,BBB,xxxi​)=λi​。

        广义瑞利商的极大值在最大广义特征值处取得，极小值在最小广义特征值处取得。下面线性判别分析的优化目标即是广义瑞利商。关于广义瑞利商的详细证明，请阅读：[广义瑞利商理论](https://www.zhihu.com/question/42041864/answer/1542605126)

#### 2 二类 LDA 原理

        假设数据集为：  
D a t a ： { ( x 1 , y 1 ) , ( x 2 , y 2 ) , ⋯   , ( x N , y N ) } x i ∈ R p , y i ∈ { + 1 , − 1 } , i = 1 , 2 , ⋯   , N (2-1) Data：\{(\pmb{x_1}, y_1), (\pmb{x_2}, y_2), \cdots, (\pmb{x_N}, y_N)\} \quad \pmb{x}_{i}\in \mathbb{R}^{p},y_{i}\in \{+1, -1\},i=1,2,\cdots ,N \tag{2-1} Data：{(x1​​x1​​​x1​,y1​),(x2​​x2​​​x2​,y2​),⋯,(xN​​xN​​​xN​,yN​)}xxxi​∈Rp,yi​∈{+1,−1},i=1,2,⋯,N(2-1)  
        分别用矩阵表示：  
X = [ x 1 , x 2 , ⋯   , x N ] N × p T = [ x 1 T x 2 T ⋮ x N T ] N × p = [ x 11 x 12 ⋯ x 1 p x 21 x 22 ⋯ x 2 p ⋮ ⋮ ⋮ ⋮ x N 1 x N 2 ⋯ x N p ] N × p (2-2) \pmb{X} = [\pmb{x}_{1},\pmb{x}_{2},\cdots ,\pmb{x}_{N}]^{T}_{N \times p}=

$$\begin{bmatrix} \pmb{x}_{1}^{T}\\ \pmb{x}_{2}^{T}\\ \vdots \\ \pmb{x}_{N}^{T} \end{bmatrix}$$

_{N \times p} =

$$\begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1p} \\ x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots &\vdots \\ x_{N1} & x_{N2} & \cdots & x_{Np} \\ \end{bmatrix}$$

_{N \times p} \tag{2-2}

XXX=[xxx1​,xxx2​,⋯,xxxN​]N×pT​=⎣⎢⎢⎢⎡​xxx1T​xxx2T​⋮xxxNT​​⎦⎥⎥⎥⎤​N×p​=⎣⎢⎢⎢⎡​x11​x21​⋮xN1​​x12​x22​⋮xN2​​⋯⋯⋮⋯​x1p​x2p​⋮xNp​​⎦⎥⎥⎥⎤​N×p​(2-2)

Y = ( y 1 y 2 ⋮ y N ) N × 1 (2-3) \pmb{Y}=

$$\begin{pmatrix} y_1\\y_2\\\vdots\\y_N \end{pmatrix}$$

_{N\times 1}\tag{2-3}

YYY=⎝⎜⎜⎜⎛​y1​y2​⋮yN​​⎠⎟⎟⎟⎞​N×1​(2-3)

        把上面的数据按 y i y_i yi​ 的取值分为两类，如下：  
{ ( x i , y i ) } i = 1 N , x i ∈ R p , y i ∈ { + 1 ‾ C 1 , − 1 ‾ C 2 } x C 1 = { x i ∣ y i = + 1 } , x C 2 = { x i ∣ y i = − 1 } ∣ x C 1 ∣ = N 1 , ∣ x C 2 ∣ = N 2 , N 1 + N 2 = N (2-4) \left \{(\pmb{x}_{i},y_{i})\right \}_{i=1}^{N},\pmb{x}_{i}\in \mathbb{R}^{p},y_{i}\in \{\underset{C_{1}}{\underline{+1}},\underset{C_{2}}{\underline{-1}}\}\\ \pmb{x}_{C_{1}}=\left \{\pmb{x}_{i}|y_{i}=+1\right \},\pmb{x}_{C_{2}}=\left \{\pmb{x}_{i}|y_{i}=-1\right \}\\ \left | \pmb{x}_{C_{1}}\right |=N_{1},\left | \pmb{x}_{C_{2}}\right |=N_{2},N_{1}+N_{2}=N \tag{2-4} {(xxxi​,yi​)}i=1N​,xxxi​∈Rp,yi​∈{C1​+1​​,C2​−1​​}xxxC1​​={xxxi​∣yi​=+1},xxxC2​​={xxxi​∣yi​=−1}∣xxxC1​​∣=N1​,∣xxxC2​​∣=N2​,N1​+N2​=N(2-4)

        上面数据矩阵的解释：数据 X \pmb{X} XXX 中有 N N N 个样本，每个样本 x i \pmb{x}_i xxxi​ 为 p p p 维数据（含有 p p p 个`feature`），数据 Y \pmb{Y} YYY 表示有 N N N 个输出，即每个样本对应一个输出（可以理解为对每个样本的标签）。这里定义 N j ( j = 0 , 1 ) N_j(j=0,1) Nj​(j=0,1) 为 C j C_j Cj​ 类样本的个数， X j ( j = 0 , 1 ) X_j(j=0,1) Xj​(j=0,1) 为 C j C_j Cj​ 类样本的集合，而 x c j ‾ ( j = 0 , 1 ) \overline{\pmb{x}_{c_j}}(j=0,1) xxxcj​​​(j=0,1) 为第 C j C_j Cj​ 类样本的均值向量，定义 Σ j ( j = 0 , 1 ) \pmb{\Sigma}_j(j=0,1) ΣΣΣj​(j=0,1) 为 C j C_j Cj​ 类样本的[协方差矩阵](https://blog.csdn.net/faceRec/article/details/1697362)。

        由于是两类数据，因此我们只需要找一条直线（方向为 w \pmb{w} www）来做投影，然后寻找最能使样本点分离的直线。当样本是二维的时候，如下图：  
![](https://img-blog.csdnimg.cn/d2c6dbe640e24bf4827ea70920451b53.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)

        从直观上来看，右图比较好，可以很好地将不同类别的样本点分离。假设投影轴的方向向量为 w \pmb{w} www，将样本点往该轴上投影以后的值 z i z_{i} zi​ 为 w T x i \pmb{w}^{T}\pmb{x}_{i} wwwTxxxi​，其中 w ∈ R p \pmb{w} \in \mathbb{R}^p www∈Rp，均值和方差按照如下方法计算：

每 类 样 例 的 均 值 ： x c j ‾ = 1 N j ∑ x ∈ X j x 投 影 后 的 样 本 点 均 值 为 ： z c j ‾ = 1 N j ∑ i = 1 N j z i = 1 N j ∑ i = 1 N j w T x i = w T x c j ‾ (2-5) 每类样例的均值： \overline{\pmb{x}_{c_j}} = \frac{1}{N_j}\sum_{\boldsymbol{x} \in X_j}\pmb{x}\\ 投影后的样本点均值为：\overline{z_{c_j}}=\frac{1}{N_j}\sum_{i=1}^{N_j}z_{i}=\frac{1}{N_j}\sum_{i=1}^{N_j}\pmb{w}^{T}\pmb{x}_{i} =\pmb{w}^T\overline{\pmb{x}_{c_j}} \tag{2-5} 每类样例的均值：xxxcj​​​=Nj​1​x∈Xj​∑​xxx 投影后的样本点均值为：zcj​​​=Nj​1​i=1∑Nj​​zi​=Nj​1​i=1∑Nj​​wwwTxxxi​=wwwTxxxcj​​​(2-5)

        对第一点，相同类内部的样本更为接近，我们假设属于两类的试验样本数量分别是 N 1 N_1 N1​和 N 2 N_2 N2​，那么我们采用方差矩阵来表征每一个类内的总体分布，这里我们使用了协方差的定义，用 Σ \pmb{\Sigma} ΣΣΣ 表示原数据的协方差：  
C 1 ： D z [ C 1 ] = 1 N 1 ∑ i = 1 N 1 ( z i − z c 1 ‾ ) ( z i − z c 1 ‾ ) T = 1 N 1 ∑ i = 1 N 1 ( w T x i − w T x c 1 ‾ ) ( w T x i − w T x c 1 ‾ ) T = w T 1 N 1 ∑ i = 1 N 1 ( x i − x c 1 ‾ ) ( x i − x c 1 ‾ ) T w = w T Σ 1 w C 2 ： D z [ C 2 ] = 1 N 2 ∑ i = 1 N 2 ( z i − z c 2 ‾ ) ( z i − z c 2 ‾ ) T = w T Σ 2 w (2-6)

$$\begin{aligned} C_1：D_z[C_1]&=\frac{1}{N_1}\sum\limits_{i=1}^{N_1}(z_i-\overline{z_{c_1}})(z_i-\overline{z_{c_1}})^T \\ &=\frac{1}{N_1}\sum\limits_{i=1}^{N_1}(\pmb{w}^T\pmb{x}_i-\pmb{w}^T\overline{\pmb{x}_{c_1}})(\pmb{w}^T\pmb{x}_i-\pmb{w}^T\overline{\pmb{x}_{c_1}})^T \\ &=\pmb{w}^T\frac{1}{N_1}\sum\limits_{i=1}^{N_1}(\pmb{x}_i-\overline{\pmb{x}_{c_1}})(\pmb{x}_i-\overline{\pmb{x}_{c_1}})^T\pmb{w} \\ &=\pmb{w}^T\pmb{\Sigma}_1\pmb{w}\\ C_2：D_z[C_2]&=\frac{1}{N_2}\sum\limits_{i=1}^{N_2}(z_i-\overline{z_{c_2}})(z_i-\overline{z_{c_2}})^T \\ &=\pmb{w}^T\pmb{\Sigma}_2\pmb{w} \end{aligned}$$

\tag{2-6}

C1​：Dz​[C1​]C2​：Dz​[C2​]​=N1​1​i=1∑N1​​(zi​−zc1​​​)(zi​−zc1​​​)T=N1​1​i=1∑N1​​(wwwTxxxi​−wwwTxxxc1​​​)(wwwTxxxi​−wwwTxxxc1​​​)T=wwwTN1​1​i=1∑N1​​(xxxi​−xxxc1​​​)(xxxi​−xxxc1​​​)Twww=wwwTΣΣΣ1​www=N2​1​i=1∑N2​​(zi​−zc2​​​)(zi​−zc2​​​)T=wwwTΣΣΣ2​www​(2-6)

所以类内距离可以记为：

D z [ C 1 ] + D z [ C 2 ] = w T ( Σ 1 + Σ 2 ) w (2-7) 

$$\begin{aligned} D_z[C_1]+D_z[C_2]=\pmb{w}^T(\pmb{\Sigma}_1+\pmb{\Sigma}_2)\pmb{w} \end{aligned}$$

 \tag{2-7} Dz​[C1​]+Dz​[C2​]=wwwT(ΣΣΣ1​+ΣΣΣ2​)www​(2-7)

对于第二点，让不同类别的数据尽可能地远离，我们可以用两类的均值表示这个距离：

( z c 1 ‾ − z c 2 ‾ ) 2 = ( 1 N 1 ∑ i = 1 N 1 w T x i − 1 N 2 ∑ i = 1 N 2 w T x i ) 2 = ( w T ( x c 1 ‾ − x c 2 ‾ ) ) 2 = w T ( x c 1 ‾ − x c 2 ‾ ) ( x c 1 ‾ − x c 2 ‾ ) T w (2-8) 

$$\begin{aligned} (\overline{z_{c_1}}-\overline{z_{c_2}})^2&=(\frac{1}{N_1}\sum\limits_{i=1}^{N_1}\pmb{w}^T\pmb{x}_i-\frac{1}{N_2}\sum\limits_{i=1}^{N_2}\pmb{w}^T\pmb{x}_i)^2 \\ &=(\pmb{w}^T(\overline{\pmb{x}_{c_1}}-\overline{\pmb{x}_{c_2}}))^2 \\ &=\pmb{w}^T(\overline{\pmb{x}_{c_1}}-\overline{\pmb{x}_{c_2}})(\overline{\pmb{x}_{c_1}}-\overline{\pmb{x}_{c_2}})^T\pmb{w} \end{aligned}$$

 \tag{2-8} (zc1​​​−zc2​​​)2​=(N1​1​i=1∑N1​​wwwTxxxi​−N2​1​i=1∑N2​​wwwTxxxi​)2=(wwwT(xxxc1​​​−xxxc2​​​))2=wwwT(xxxc1​​​−xxxc2​​​)(xxxc1​​​−xxxc2​​​)Twww​(2-8)

        由于 LDA 需要让不同类别的数据的类别中心之间的距离尽可能的大，也就是我们要最大化 ( z c 1 ‾ − z c 2 ‾ ) 2 (\overline{z_{c1}}-\overline{z_{c2}})^2 (zc1​​−zc2​​)2，同时我们希望同一种类别数据的投影点尽可能的接近，也就是要同类样本投影点的协方差 D z [ C 1 ] D_z[C_1] Dz​[C1​] 和 D z [ C 2 ] D_z[C_2] Dz​[C2​] 尽可能的小，即最小化 D z [ C 1 ] + D z [ C 2 ] D_z[C_1]+D_z[C_2] Dz​[C1​]+Dz​[C2​]。综上所述，由于协方差是一个矩阵，于是我们用将这两个值相除来得到我们的损失函数，并最大化这个值，即我们的优化目标为：  
a r g m a x w J (w) = a r g m a x w ( z c 1 ‾ − z c 2 ‾ ) 2 D z [ C 1 ] + D z [ C 2 ] = a r g m a x w w T ( x c 1 ‾ − x c 2 ‾ ) ( x c 1 ‾ − x c 2 ‾ ) T w w T ( Σ 1 + Σ 2 ) w (2-9)

$$\begin{aligned} \mathop{argmax}\limits_\boldsymbol{w}J(\pmb{w})&=\mathop{argmax}\limits_\boldsymbol{w}\frac{(\overline{z_{c1}}-\overline{z_{c2}})^2}{D_z[C_1]+D_z[C_2]} \\ &=\mathop{argmax}\limits_\boldsymbol{w}\frac{w^T(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}})(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}})^T\boldsymbol{w}}{\boldsymbol{w}^T(\boldsymbol{\Sigma}_1+\boldsymbol{\Sigma}_2)\boldsymbol{w}} \end{aligned}$$

\tag{2-9}

wargmax​J(www)​=wargmax​Dz​[C1​]+Dz​[C2​](zc1​​−zc2​​)2​=wargmax​wT(Σ1​+Σ2​)wwT(xc1​​−xc2​​)(xc1​​−xc2​​)Tw​​(2-9)

        我们一般定义类内散度矩阵 S w \pmb{S}_w SSSw​ 为：  
S w = Σ 1 + Σ 2 (2-10) \pmb{S}_w=\boldsymbol{\Sigma}_1+\boldsymbol{\Sigma}_2 \tag{2-10} SSSw​=Σ1​+Σ2​(2-10)

        同时定义类间散度矩阵 S b \pmb{S}_b SSSb​ 为：

S b = ( x c 1 ‾ − x c 2 ‾ ) ( x c 1 ‾ − x c 2 ‾ ) T (2-11) \pmb{S}_b=(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}})(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}})^T \tag{2-11} SSSb​=(xc1​​−xc2​​)(xc1​​−xc2​​)T(2-11)

        所以，优化目标重写为：  
a r g m a x w J (w) = a r g m a x w w T S b w w T S w w (2-12) \mathop{argmax}\limits_\boldsymbol{w}J(\pmb{w}) =\mathop{argmax}\limits_\boldsymbol{w}\frac{\boldsymbol{w}^T\boldsymbol{S}_b\boldsymbol{w}}{\boldsymbol{w}^T\boldsymbol{S}_w\boldsymbol{w}} \tag{2-12} wargmax​J(www)=wargmax​wTSw​wwTSb​w​(2-12)

        这样，我们就把损失函数和原数据集以及参数结合起来了。

**方法一：**  
        观察式子（2-12）不正是我们的广义瑞利商的形式，所以利用第一节讲到的广义瑞利商的性质，可以得到 J ( w ^ ) J(\hat{\pmb{w}}) J(www^) 最大值为矩阵 S w − 1 2 S b S w − 1 2 \boldsymbol{S}^{−\frac{1}{2}}_w\boldsymbol{S}_b\boldsymbol{S}^{−\frac{1}{2}}_w Sw−21​​Sb​Sw−21​​ 的最大特征值，而对应的 w ^ \hat{\pmb{w}} www^ 为 S w − 1 2 S b S w − 1 2 \boldsymbol{S}^{−\frac{1}{2}}_w\boldsymbol{S}_b\boldsymbol{S}^{−\frac{1}{2}}_w Sw−21​​Sb​Sw−21​​ 的最大特征值对应的特征向量! 而 S w − 1 S b \pmb{S}^{−1}_w\pmb{S}_b SSSw−1​SSSb​ 的特征值和 S w − 1 2 S b S w − 1 2 \boldsymbol{S}^{−\frac{1}{2}}_w\boldsymbol{S}_b\boldsymbol{S}^{−\frac{1}{2}}_w Sw−21​​Sb​Sw−21​​ 的特征值相同， S w − 1 S b \pmb{S}^{−1}_w\pmb{S}_b SSSw−1​SSSb​ 的特征向量 w \pmb{w} www 和 S w − 1 2 S b S w − 1 2 \boldsymbol{S}^{−\frac{1}{2}}_w\boldsymbol{S}_b\boldsymbol{S}^{−\frac{1}{2}}_w Sw−21​​Sb​Sw−21​​ 的特征向量 w ^ \hat{\pmb{w}} www^ 满足 w = S w − 1 2 w ^ \pmb{w}=S^{−\frac{1}{2}}_{w}\hat{\pmb{w}} www=Sw−21​​www^ 的关系!

**方法二：**  
        对于两类的情况，可以直接对这个损失函数求偏导，注意我们其实对 w \pmb{w} www 的绝对值没有任何要求，只对方向有要求，因此只要一个方程就可以求解了：  
∂ J (w) ∂ w = 2 S b w ( w T S w w ) − 1 + w T S b w ( − 1 ) ( w T S w w ) − 2 2 S b w = 2 [ S b w ( w T S w w ) − 1 + w T S b w ( − 1 ) ( w T S w w ) − 2 S b w ] = 0 （ w T S b w , w T S w w ∈ R ） ⇒ S b w ( w T S w w ) − w T S b w S w w = 0 ⇒ w T S b w S w w = S b w ( w T S w w ) ⇒ S w w = w T S w w w T S b w S b w ⇒ w = w T S w w w T S b w S w − 1 S b w （ 要 注 意 对 于 w 我 们 只 关 注 它 的 方 向 ， 不 关 心 它 的 大 小 。 ） ⇒ w ∝ S w − 1 S b w ⇒ w ∝ S w − 1 ( x c 1 ‾ − x c 2 ‾ ) ( x c 1 ‾ − x c 2 ‾ ) T w ⏟ 1 维 ⇒ w ∝ S w − 1 ( x c 1 ‾ − x c 2 ‾ ) ⇒ w ∝ ( S C 1 + S C 2 ) − 1 ( x c 1 ‾ − x c 2 ‾ ) (2-13) \frac{\partial J(\boldsymbol{w})}{\partial \boldsymbol{w}}=2\boldsymbol{S}_{b}\boldsymbol{w}(\boldsymbol{w}^{T}\boldsymbol{S}_{w}\boldsymbol{w})^{-1}+\boldsymbol{w}^{T}\boldsymbol{S}_{b}\boldsymbol{w}(-1)(\boldsymbol{w}^{T}\boldsymbol{S}_{w}\boldsymbol{w})^{-2}2\boldsymbol{S}_{b}\boldsymbol{w}\\ =2[\boldsymbol{S}_{b}\boldsymbol{w}(\boldsymbol{w}^{T}\boldsymbol{S}_{w}\boldsymbol{w})^{-1}+\boldsymbol{w}^{T}\boldsymbol{S}_{b}\boldsymbol{w}(-1)(\boldsymbol{w}^{T}\boldsymbol{S}_{w}\boldsymbol{w})^{-2}\boldsymbol{S}_{b}\boldsymbol{w}]=0\\ （\boldsymbol{w}^{T}\boldsymbol{S}_{b}\boldsymbol{w},\boldsymbol{w}^{T}\boldsymbol{S}_{w}\boldsymbol{w}\in \mathbb{R}）\\ \Rightarrow \boldsymbol{S}_{b}\boldsymbol{w}(\boldsymbol{w}^{T}\boldsymbol{S}_{w}\boldsymbol{w})-\boldsymbol{w}^{T}\boldsymbol{S}_{b}\boldsymbol{w}\boldsymbol{S}_{w}\boldsymbol{w}=0\\ \Rightarrow \boldsymbol{w}^{T}\boldsymbol{S}_{b}\boldsymbol{w}\boldsymbol{S}_{w}\boldsymbol{w}=\boldsymbol{S}_{b}\boldsymbol{w}(\boldsymbol{w}^{T}\boldsymbol{S}_{w}\boldsymbol{w})\\ \Rightarrow \boldsymbol{S}_{w}\boldsymbol{w}=\frac{\boldsymbol{w}^{T}\boldsymbol{S}_{w}\boldsymbol{w}}{\boldsymbol{w}^{T}\boldsymbol{S}_{b}\boldsymbol{w}}\boldsymbol{S}_{b}\boldsymbol{w}\\ \Rightarrow \boldsymbol{w}=\frac{\boldsymbol{w}^{T}\boldsymbol{S}_{w}\boldsymbol{w}}{\boldsymbol{w}^{T}\boldsymbol{S}_{b}\boldsymbol{w}}\boldsymbol{S}_{w}^{-1}\boldsymbol{S}_{b}\boldsymbol{w}\\ （要注意对于 \ boldsymbol{w} 我们只关注它的方向，不关心它的大小。）\\ \Rightarrow \boldsymbol{w}\propto \boldsymbol{S}_{w}^{-1}\boldsymbol{S}_{b}\boldsymbol{w}\\ \Rightarrow \boldsymbol{w}\propto \boldsymbol{S}_{w}^{-1}(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}})\underset{1 维}{\underbrace{(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}})^{T}\boldsymbol{w}}}\\ \Rightarrow \boldsymbol{w}\propto \boldsymbol{S}_{w}^{-1}(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}})\\ \Rightarrow \boldsymbol{w}\propto (\boldsymbol{S}_{C_{1}}+\boldsymbol{S}_{C_{2}})^{-1}(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}}) \tag{2-13} ∂w∂J(w)​=2Sb​w(wTSw​w)−1+wTSb​w(−1)(wTSw​w)−22Sb​w=2[Sb​w(wTSw​w)−1+wTSb​w(−1)(wTSw​w)−2Sb​w]=0（wTSb​w,wTSw​w∈R）⇒Sb​w(wTSw​w)−wTSb​wSw​w=0⇒wTSb​wSw​w=Sb​w(wTSw​w)⇒Sw​w=wTSb​wwTSw​w​Sb​w⇒w=wTSb​wwTSw​w​Sw−1​Sb​w（要注意对于 w 我们只关注它的方向，不关心它的大小。）⇒w∝Sw−1​Sb​w⇒w∝Sw−1​(xc1​​−xc2​​)1 维 (xc1​​−xc2​​)Tw​​⇒w∝Sw−1​(xc1​​−xc2​​)⇒w∝(SC1​​+SC2​​)−1(xc1​​−xc2​​)(2-13)

        于是 S w − 1 ( x c 1 ‾ − x c 2 ‾ ) \boldsymbol{S}_w^{-1}(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}}) Sw−1​(xc1​​−xc2​​) 就是我们需要寻找的方向。最后可以归一化求得单位的 w \boldsymbol{w} w 值。

        对于二类的时候， S b w \pmb{S}_b\pmb{w} SSSb​www 的方向恒平行于 ( x c 1 ‾ − x c 2 ‾ ) (\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}}) (xc1​​−xc2​​)，不妨令 S b w = ( x c 1 ‾ − x c 2 ‾ ) ∗ λ ^ \pmb{S}_b\pmb{w} =(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}})*\hat{\lambda} SSSb​www=(xc1​​−xc2​​)∗λ^，将其带入： ( S w − 1 S b ) w = S w − 1 ( x c 1 ‾ − x c 2 ‾ ) ∗ λ ^ = λ w (\pmb{S}^{−1}_w\pmb{S}_b)\pmb{w}=\pmb{S}^{−1}_w(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}})*\hat{\lambda}=\lambda\pmb{w} (SSSw−1​SSSb​)www=SSSw−1​(xc1​​−xc2​​)∗λ^=λwww， 由于对 w \pmb{w} www 扩大缩小任何倍不影响结果，因此可以约去两边的未知常数 λ \lambda λ 和 λ ^ \hat{\lambda} λ^，可以得到 w = S w − 1 ( x c 1 ‾ − x c 2 ‾ ) \pmb{w}=\pmb{S}^{−1}_w(\overline{\boldsymbol{x}_{c1}}-\overline{\boldsymbol{x}_{c2}}) www=SSSw−1​(xc1​​−xc2​​)， 也就是说我们只要求出原始二类样本的均值和方差就可以确定最佳的投影方向 w \pmb{w} www 了。

**方法三：**  
        在我们求导之前，需要对分母进行归一化，因为不做归一的话， w \pmb{w} www 扩大任何倍，都成立，我们就无法确定 w \pmb{w} www。因此我们令 w T S w w = 1 {\pmb{w}^T\pmb{S}_w\pmb{w}}=1 wwwTSSSw​www=1（这里上下式子一同缩放不影响最后求 a r g m a x w J (w) \mathop{argmax}\limits_\boldsymbol{w}J(\pmb{w}) wargmax​J(www) 的结果），这样使用拉格朗日乘子法就可以得到：  
J (w) = w T S b w − λ ( w T S w w − 1 ) ∂ ∂ w J ( w ) = 2 S b w − 2 λ S w w = 0 S b w = λ S w w (2-14) J(\boldsymbol{w})=\boldsymbol{w}^T\boldsymbol{S}_b\boldsymbol{w}-\lambda(\boldsymbol{w}^T\boldsymbol{S}_w\boldsymbol{w}-1) \\ \frac{\partial}{\partial \boldsymbol{w}}J(\boldsymbol{w})= 2\boldsymbol{S}_b\boldsymbol{w}-2\lambda\boldsymbol{S}_w\boldsymbol{w}=0\\ \boldsymbol{S}_b\boldsymbol{w}=\lambda\boldsymbol{S}_w\boldsymbol{w} \tag{2-14} J(w)=wTSb​w−λ(wTSw​w−1)∂w∂​J(w)=2Sb​w−2λSw​w=0Sb​w=λSw​w(2-14)

        如果 S w \boldsymbol{S}_w Sw​ 可逆，就又变回了求特征向量的套路：  
S w − 1 S b w = λ w (2-15) \boldsymbol{S}_w^{-1}\boldsymbol{S}_b\boldsymbol{w}=\lambda\boldsymbol{w}\tag{2-15} Sw−1​Sb​w=λw(2-15)

        当然，在实际问题中， 常出现 S w \boldsymbol{S}_w Sw​ 不可逆的问题，不过一般情况下可以通过其他方法解决：

*   令 S w = S w + γ I \boldsymbol{S}_w=\boldsymbol{S}_w+\gamma \pmb{I} Sw​=Sw​+γIII，其中 γ \gamma γ 是一个特别小的数，这样 S w \boldsymbol{S}_w Sw​ 一定可逆
    
*   先使用 PCA 对数据进行降维，使得在降维后的数据上 S w \boldsymbol{S}_w Sw​ 可逆，再使用 LDA
    

#### 3 多类 LDA 原理

        有了二类 LDA 的基础，我们再来看看多类别 LDA 的原理。

        假设我们的数据集 D = { ( x 1 , y 1 ) , ( x 2 , y 2 ) , . . . , ( x N , y N ) } D=\{(\pmb{x}_1,y_1), (\pmb{x}_2,y_2), ...,(\pmb{x}_N,y_N)\} D={(xxx1​,y1​),(xxx2​,y2​),...,(xxxN​,yN​)}，其中任意样本 x i ∈ R p \pmb{x}_i\in\mathbb{R}^p xxxi​∈Rp 为 p p p 维向量， y i ∈ { C 1 , C 2 , ⋯   , C k } y_i\in\{C_1,C_2,\cdots,C_k\} yi​∈{C1​,C2​,⋯,Ck​}。我们定义 N j ( j = 1 , 2 , ⋯   , k ) N_j(j=1,2,\cdots,k) Nj​(j=1,2,⋯,k) 为第 j j j 类样本的个数， X j ( j = 1 , 2 , ⋯   , k ) X_j(j=1,2,\cdots,k) Xj​(j=1,2,⋯,k) 为第 j j j 类样本的集合，而 x c j ‾ ( j = 1 , 2 , ⋯   , k ) \overline{\boldsymbol{x}_{c_j}}(j=1,2,\cdots,k) xcj​​​(j=1,2,⋯,k) 为第 j j j 类样本的均值向量，定义 Σ j ( j = 1 , 2... k ) \pmb{\Sigma}_j(j=1,2...k) ΣΣΣj​(j=1,2...k) 为第 j j j 类样本的协方差矩阵。在二类 LDA 里面定义的公式可以很容易的类推到多类 LDA。

        由于我们是多类向低维投影，则此时投影到的低维空间就不是一条直线，而是一个超平面了。假设我们投影到的低维空间的维度为 q q q，对应的基向量为 ( w 1 , w 2 , ⋯   , w q ) (\pmb{w}_1,\pmb{w}_2,\cdots,\pmb{w}_q) (www1​,www2​,⋯,wwwq​)，基向量组成的矩阵为 W \pmb{W} WWW, 它是一个 p × q p \times q p×q 的矩阵。

        此时我们的优化目标应该可以变成为：  
W T S b W W T S w W (3-1) \frac{\boldsymbol{W}^T\boldsymbol{S}_b\boldsymbol{W}}{\boldsymbol{W}^T\boldsymbol{S}_w\boldsymbol{W}} \tag{3-1} WTSw​WWTSb​W​(3-1)

        其中 S b = ∑ j = 1 k N j ( x c j ‾ − x ‾ ) ( x c j ‾ − x ‾ ) T \pmb{S}_b = \sum\limits_{j=1}^{k}N_j(\overline{\boldsymbol{x}_{c_j}}-\overline{\boldsymbol{x}})(\overline{\boldsymbol{x}_{c_j}}-\overline{\boldsymbol{x}})^T SSSb​=j=1∑k​Nj​(xcj​​​−x)(xcj​​​−x)T， x ‾ \overline{\boldsymbol{x}} x 为所有样本均值向量。 S w = ∑ j = 1 k S w j = ∑ j = 1 k ∑ x ∈ X j ( x − x c j ‾ ) ( x − x c j ‾ ) T \pmb{S}_w = \sum\limits_{j=1}^{k}\pmb{S}_{wj} = \sum\limits_{j=1}^{k}\sum\limits_{\boldsymbol{x} \in \boldsymbol{X}_j}(\pmb{x}-\overline{\boldsymbol{x}_{c_j}})(\pmb{x}-\overline{\boldsymbol{x}_{c_j}})^T SSSw​=j=1∑k​SSSwj​=j=1∑k​x∈Xj​∑​(xxx−xcj​​​)(xxx−xcj​​​)T

        这里的 W T S b W \boldsymbol{W}^T\boldsymbol{S}_b\boldsymbol{W} WTSb​W 和 W T S w W \boldsymbol{W}^T\boldsymbol{S}_w\boldsymbol{W} WTSw​W 都是矩阵，不是标量，无法作为一个标量函数来优化！也就是说，我们无法直接用二类 LDA 的优化方法，怎么办呢？一般来说，我们可以用其他的一些替代优化目标来实现。

        常见的一个 LDA 多类优化目标函数定义为：  
a r g    m a x ⏟ W      J (W) = ∏ d i a g W T S b W ∏ d i a g W T S w W (3-2) \underbrace{arg\;max}_\boldsymbol{W}\;\;J(\boldsymbol{W}) = \frac{\prod\limits_{diag}\boldsymbol{W}^T\boldsymbol{S}_b\boldsymbol{W}}{\prod\limits_{diag}\boldsymbol{W}^T\boldsymbol{S}_w\boldsymbol{W}}\tag{3-2} W argmax​​J(W)=diag∏​WTSw​Wdiag∏​WTSb​W​(3-2)

        其中 ∏ d i a g A \prod\limits_{diag}\boldsymbol{A} diag∏​A 为 A \boldsymbol{A} A 的主对角线元素的乘积， W \boldsymbol{W} W 为 p × q p \times q p×q 的矩阵。

         J (W) J(\pmb{W}) J(WWW) 的优化过程可以转化为：  
J (W) = ∏ i = 1 d w i T S b w i ∏ i = 1 d w i T S w w i = ∏ i = 1 d w i T S b w i w i T S w w i (3-3) J(\boldsymbol{W}) = \frac{\prod\limits_{i=1}^d\boldsymbol{w}_i^T\boldsymbol{S}_b\boldsymbol{w}_i}{\prod\limits_{i=1}^d\boldsymbol{w}_i^T\boldsymbol{S}_w\boldsymbol{w}_i} = \prod\limits_{i=1}^d\frac{\boldsymbol{w}_i^T\boldsymbol{S}_b\boldsymbol{w}_i}{\boldsymbol{w}_i^T\boldsymbol{S}_w\boldsymbol{w}_i}\tag{3-3} J(W)=i=1∏d​wiT​Sw​wi​i=1∏d​wiT​Sb​wi​​=i=1∏d​wiT​Sw​wi​wiT​Sb​wi​​(3-3)

        仔细观察上式最右边，这不就是广义瑞利商嘛！最大值是矩阵 S w − 1 S b \pmb{S}^{−1}_w\pmb{S}_b SSSw−1​SSSb​ 的最大特征值，最大的 q q q 个值的乘积就是矩阵 S w − 1 S b \pmb{S}^{−1}_w\pmb{S}_b SSSw−1​SSSb​ 的最大的 q q q 个特征值的乘积，此时对应的矩阵 W \pmb{W} WWW 为这最大的 q q q 个特征值对应的特征向量张成的矩阵。

>         由于 W \pmb{W} WWW 是一个利用了样本的类别得到的投影矩阵，因此它的降维到的维度 q q q 最大值为 k − 1 k-1 k−1。为什么最大维度不是类别数 k k k 呢？因为 S b \pmb{S}_b SSSb​ 中每个 ( x c j ‾ − x ‾ ) (\overline{\boldsymbol{x}_{c_j}}-\overline{\boldsymbol{x}}) (xcj​​​−x) 的秩为 1，因此协方差矩阵相加后最大的秩为 k k k (矩阵的秩小于等于各个相加矩阵的秩的和)，但是由于如果我们知道前 k − 1 k-1 k−1 个 x c j ‾ \overline{\boldsymbol{x}_{c_j}} xcj​​​ 后，最后一 个 x c k ‾ \overline{\boldsymbol{x}_{c_k}} xck​​​ 可以由前 k − 1 k-1 k−1 个 x c j ‾ \overline{\boldsymbol{x}_{c_j}} xcj​​​ 线性表示，因此 S b \pmb{S}_b SSSb​ 的秩最大为 k − 1 k-1 k−1，即特征向量最多有 k − 1 k-1 k−1 个。

#### 4 LDA 算法流程

        在第三节和第四节我们讲述了 LDA 的原理，现在我们对 LDA 降维的流程做一个总结。

> 输入：数据集 D = { ( x 1 , y 1 ) , ( x 2 , y 2 ) , . . . , ( x N , y N ) } D=\{(\pmb{x}_1,y_1), (\pmb{x}_2,y_2), ...,(\pmb{x}_N,y_N)\} D={(xxx1​,y1​),(xxx2​,y2​),...,(xxxN​,yN​)}，其中任意样本 x i ∈ R p \pmb{x}_i\in\mathbb{R}^p xxxi​∈Rp 为 p p p 维向量， y i ∈ { C 1 , C 2 , ⋯   , C k } y_i\in\{C_1,C_2,\cdots,C_k\} yi​∈{C1​,C2​,⋯,Ck​}，降维到的维度 q q q。
> 
> 输出：降维后的样本集 D ′ D′ D′
> 
> 1.  计算类内散度矩阵 S w \pmb{S}_w SSSw​
>     
> 2.  计算类间散度矩阵 S b \pmb{S}_b SSSb​
>     
> 3.  计算矩阵 S w − 1 S b \pmb{S}^{−1}_w\pmb{S}_b SSSw−1​SSSb​
>     
> 4.  计算 S w − 1 S b \pmb{S}^{−1}_w\pmb{S}_b SSSw−1​SSSb​ 的最大的 q q q 个特征值和对应的 q q q 个特征向量 ( w 1 , w 2 , ⋯   , w d ) (\pmb{w}_1,\pmb{w}_2,\cdots,\pmb{w}_d) (www1​,www2​,⋯,wwwd​) 得到投影矩阵 W \pmb{W} WWW
>     
> 5.  对样本集中的每一个样本特征 x i \pmb{x}_i xxxi​，转化为新的样本 z i = W T x i \pmb{z}_i = \pmb{W}^T\pmb{x}_i zzzi​=WWWTxxxi​
>     
> 6.  得到输出样本集 D ′ = ( z 1 , y 1 ) , ( z 2 , y 2 ) , ⋯   , ( ( z m , y m ) ) D′={(\pmb{z}_1,y_1),(\pmb{z}_2,y_2),\cdots,((\pmb{z}_m,y_m))} D′=(zzz1​,y1​),(zzz2​,y2​),⋯,((zzzm​,ym​))
>     

        以上就是使用 LDA 进行降维的算法流程。实际上 LDA 除了可以用于降维以外，还可以用于分类。一个常见的 LDA 分类基本思想是假设各个类别的样本数据符合高斯分布，这样利用 LDA 进行投影后，可以利用极大似然估计计算各个类别投影数据的均值和方差，进而得到该类别高斯分布的概率密度函数。当一个新的样本到来后，我们可以将它投影，然后将投影后的样本特征分别带入各个类别的高斯分布概率密度函数，计算它属于这个类别的概率，最大的概率对应的类别即为预测类别。

#### 5 程序实现

##### 5.1 Python 实现

```
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def LDA(data, target, n_dim):
    '''
    :param data: (n_samples, n_features)
    :param target: data class
    :param n_dim: target dimension
    :return: (n_samples, n_dims)
 	'''   
    clusters = np.unique(target)

    if n_dim > len(clusters)-1:
        print("K is too much")
        print("please input again")
        exit(0)

    #within_class scatter matrix
    Sw = np.zeros((data.shape[1],data.shape[1]))
    for i in clusters:
        datai = data[target == i]
        datai = datai-datai.mean(0)
        Swi = np.mat(datai).T*np.mat(datai)
        Sw += Swi

    #between_class scatter matrix
    SB = np.zeros((data.shape[1],data.shape[1]))
    u = data.mean(0)  #所有样本的平均值
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(0)  #某个类别的平均值
        SBi = Ni*np.mat(ui - u).T*np.mat(ui - u)
        SB += SBi
    S = np.linalg.inv(Sw)*SB
    eigVals,eigVects = np.linalg.eig(S)  #求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-n_dim-1):-1]
    w = eigVects[:,eigValInd]
    data_ndim = np.dot(data, w)

    return data_ndim


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    target_names = iris.target_names
    X_r2 = LDA(X, Y, 2)
    colors = ['navy', 'turquoise', 'darkorange']

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], alpha=.8, color=color,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset by Python')

    plt.show()

```

![](https://img-blog.csdnimg.cn/04ebf032a4e241da9a6fbe38cf20fea2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)

##### 5.2 sklearn 实现

         在`scikit-learn`中， LDA 类是`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`。既可以用于分类又可以用于降维。当然，应用场景最多的还是降维。和 PCA 类似，LDA 降维基本也不用调参，只需要指定降维到的维数即可。对`LinearDiscriminantAnalysis`模块的详细介绍，可以参考下面两篇文章：[sklearn 学习 9----LDA (discriminat_analysis)](https://www.cnblogs.com/Lee-yl/p/9263687.html) 和 [用 scikit-learn 进行 LDA 降维](https://www.cnblogs.com/pinard/p/6249328.html)

```
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
X = iris.data
Y = iris.target
target_names = iris.target_names
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, Y).transform(X)
colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset by sklearn')

plt.show()

```

![](https://img-blog.csdnimg.cn/61d486df317242498f982978b2a71a42.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)

#### 6 LDA vs PCA

        LDA 用于降维，和 PCA 有很多相同，也有很多不同的地方，因此值得好好的比较一下两者的降维异同点。  
![](https://img-blog.csdnimg.cn/b6f2cb0d94fd4a8097f3baa473e87ad9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_19,color_FFFFFF,t_70,g_se,x_16#pic_center)

        LDA 最多将数据降维至 k − 1 k-1 k−1 维，也就是说如果有两类数据，最终降维只能到 1 维，也就是说投影到一个直线上。这在很多情况下无法对数据进行很好的投影，例如下图中的几种情况。也就是说，LDA 不适合对非高斯分布的样本进行降维。  
![](https://img-blog.csdnimg.cn/fcac29ee469e453a9cfaaf170d3165ce.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)

        若训练样本集两类的均值有明显的差异，LDA 降维的效果较优，如下图：

![](https://img-blog.csdnimg.cn/32d52ca6060348ad9bdb3501a0917b70.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)

        若训练样本集两类的均值无明显的差异，但协方差差异很大，PCA 降维的效果较优，如下图：

![](https://img-blog.csdnimg.cn/514af8298fbd4c9ea3011822c5ca1172.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_10,color_FFFFFF,t_70,g_se,x_16#pic_center)

#### LDA 算法小结

        LDA 算法既可以用来降维，又可以用来分类，但是目前来说，主要还是用于降维。在我们进行图像识别图像识别相关的数据分析时，LDA 是一个有力的工具。下面总结下 LDA 算法的优缺点。

        LDA 算法的主要优点有：

![](https://img-blog.csdnimg.cn/11e64aac4ef64ec881400057f4e1b445.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_10,color_FFFFFF,t_70,g_se,x_16#pic_center)

        LDA 算法的主要缺点有：

![](https://img-blog.csdnimg.cn/2ce543486ba8452c8d2e00736c00abd4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6ZW_6Lev5ryr5ryrMjAyMQ==,size_10,color_FFFFFF,t_70,g_se,x_16#pic_center)

#### 参考

*   线性判别分析 LDA 原理总结：[https://www.cnblogs.com/pinard/p/6244265.html](https://www.cnblogs.com/pinard/p/6244265.html)
*   线性判别分析（Linear Discriminant Analysis）（一）:[https://www.cnblogs.com/jerrylead/archive/2011/04/21/2024384.html](https://www.cnblogs.com/jerrylead/archive/2011/04/21/2024384.html)
*   机器学习中的数学 (4)- 线性判别分析（LDA）, 主成分分析 (PCA)：[https://www.cnblogs.com/LeftNotEasy/archive/2011/01/08/lda-and-pca-machine-learning.html](https://www.cnblogs.com/LeftNotEasy/archive/2011/01/08/lda-and-pca-machine-learning.html)
*   线性判别分析 (Linear Discriminant Analysis, LDA）算法分析：[https://blog.csdn.net/warmyellow/article/details/5454943](https://blog.csdn.net/warmyellow/article/details/5454943)
*   如何理解线性判别分类器（LDA）：[https://mp.weixin.qq.com/s/5aqJ1mdS3hzJSh1rILMt5g](https://mp.weixin.qq.com/s/5aqJ1mdS3hzJSh1rILMt5g)
*   机器学习算法原理笔记（三）—— LDA 线性判别分析：[https://www.jianshu.com/p/c365e818331b](https://www.jianshu.com/p/c365e818331b)
*   Python 机器学习笔记: 线性判别分析 (LDA）算法：[https://www.cnblogs.com/wj-1314/p/10234256.html](https://www.cnblogs.com/wj-1314/p/10234256.html)
*   瑞利商（Rayleigh Quotient）及瑞利定理（Rayleigh-Ritz theorem）的证明：[https://blog.csdn.net/klcola/article/details/104800804](https://blog.csdn.net/klcola/article/details/104800804)
*   【线性代数】瑞利商和广义瑞利商：[https://zhuanlan.zhihu.com/p/432080955](https://zhuanlan.zhihu.com/p/432080955)