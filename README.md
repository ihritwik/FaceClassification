# Face_Classification
A comparative study between 5 different binary classification techniques. In particular, this project is aimed at increasing familiarity with the **Expectation-Maximization** steps for each of the below listed algorithms:

* Single Gaussian Model
* Gaussian Mixture Model
* T-Distribution Model
* Mixture of T-Distribution
* Factor Analyzer

This section presents a brief description of the expectation maximization
(EM) algorithm. The goal is to provide just enough information to use this technique for fitting models.

## Expectation-Maximization
The EM algorithm is a general-purpose tool for fitting parameters ***θ*** in models of the form described below:

![Theta argmax][Eq-1]

To model a complex probability density function over the input variable ***x***, we will introduce a hidden or latent variable ***h***, which may be discrete or continuous. To exploit the hidden variables, we describe the final density ***Pr(x)*** as the marginalization of the joint density ***Pr(x ; h)*** between ***x*** and ***h*** so that:

![Pr x][Eq-2]

Which can also be written taking parameter ***θ*** into consideration:

![Pr xh][Eq-3]

In the E-step at iteration ***t+1*** we set each distribution ***q<sub>i</sub>(h<sub>i</sub>)*** to be the posterior distributions ***Pr(h<sub>i</sub>|x<sub>i</sub> ; θ)*** over that hidden variable given the associated data example
and the current parameters ***θ<sup>[t]</sup>***. To compute these we use Bayes’ rule:

![qh][Eq-4]

In the M-step we directly maximize the bound with respect to the parameters ***θ***. Taking the derivative and simplifying, we get:

![theta][Eq-5]

Each of these steps is guaranteed to improve the bound, and iterating them alternately is guaranteed to find at least a local maximum with respect to ***θ***.  We **initialize the parameters randomly** and alternate between performing the E and M step.

Now let's delve deeper into each of the five use cases:

## Single Gaussian Model

This is a generative classification model which assumes the entire training data follows a single Gaussian distribution.
Thus, having a single gaussian component, the parameters can be learned from the underlying data itself. The E-M algorithm is not needed for this model.

Hence, the posterior probability is given by:

![Gaussian][Eq-6]   

where ***µ<sub>w</sub>*** and ***Σ<sub>w</sub>*** are the mean and covariance for class ***w***

## Gaussian Mixture Model

This is a generative classification model which assumes the entire training data follows a multiple Gaussian distributions, i.e. a mixture of normal distributions. The mixture of Gaussians (MoG) is a prototypical example for the EM algorithm, where the data are described as a weighted sum of ***K*** normal distributions:

![GMM][Eq-7]

where ***µ<sub>1...K</sub>*** and ***Σ<sub>1...K</sub>*** are the means and covariances of the normal distributions
and ***λ<sub>1...K</sub>*** are positive valued weights that sum to one. The mixtures of Gaussians
model describes complex multi-modal probability densities by combining simpler
constituent distributions.

To learn the MoG parameters ***θ = { λ<sub>k</sub> , µ<sub>k</sub> , Σ<sub>k</sub> }<sup>K</sup><sub>k=1</sub>*** from training data ***{ x<sub>i</sub> }<sup>I</sup><sub>i=1</sub>*** we apply the EM algorithm.

In the **E-step**, we maximize the bound with respect to the distributions ***q<sub>i</sub>(h<sub>i</sub>)***
by finding the posterior probability distribution ***Pr(h<sub>i</sub>|x<sub>i</sub>)*** of each hidden variable
***h<sub>i</sub>*** given the observation ***x<sub>i</sub>*** and the current parameter settings. This is achieved by computing the probability ***Pr(h<sub>i</sub> = k|x<sub>i</sub> ; θ<sup>[t]</sup>)*** that the ***k<sup>th</sup>*** normal
distribution was responsible for the ***i<sup>th</sup>*** data point. We denote this
responsibility by ***r<sub>ik</sub>*** for short:

![GMM-E][Eq-8]

In the **M-step**, we maximize the bound with respect to the parameters ***θ = { λ<sub>k</sub> ; µ<sub>k</sub> ; Σ<sub>k</sub> }<sup>K</sup><sub>k=1</sub>***, which gives us the following closed-form update equations:

![GMM-E][Eq-8]

## T-Distribution Model
The height of the normal pdf falls off very rapidly as we move into the tails. The effect of this is that outliers (unusually extreme observations) drastically affect the estimated parameters. The T-distribution
is a closely related distribution in which the length of the tails is parameterized.

Jumping right in:

In the **E-step**, we compute the following Expectations:

![T-E][Eq-10]

where ***Ψ( [•] )*** is the digamma function.

In the **M-step**, we optimize ***µ*** and ***Σ*** using the following update equations:

![T-M][Eq-11]

## Mixture of T-Distribution Model

This is a generative model very similar to MoG and the T-DIstribution model. It employs the basic essence of MoG where the data is described as a weighted sum of ***K*** T-Distributions.

The **E-step** is similar to MoG where we compute a responsibility matrix by computing the probability ***Pr(h<sub>i</sub> = k|x<sub>i</sub> ; θ<sup>[t]</sup>)*** that the ***k<sup>th</sup>*** T-distribution was responsible for the ***i<sup>th</sup>*** data point.

The **M-step** is similar to single T-Distribution except expanded to accommodate multiple components.

## Factor Analyzer Model

As the data are 60x60 RGB images, with the full multivariate normal distribution, the covariance matrix is a high-dimensional 10800x10800 space. There are two main problems with this approach:

* Large number of training examples are neede to get a good estimate of all of these parameters
in the absence of prior information.
* Furthermore, to store the covariance matrix we will need a large amount of memory.
* Inverting this large a matrix is highly computationally expensive.

Of course, we could just use the diagonal form of the covariance matrix which contains only 10800 parameters. However, this is too great a simplification.

Factor analysis provides a compromise in which the covariance matrix is structured so that it contains fewer unknown parameters than the full matrix but more than the diagonal form. One way to think about the covariance of a factor analyzer is that it models part of the high-dimensional space with a full model and mops up remaining variation with a diagonal model.

The probability density function of a factor analyzer is given by:

![FA][Eq-12]

In the **E-step**, We extract the following expectations from the posterior distribution:

![FA-E][Eq-13]

In the **M-step**, we have the following closed-form update equations:

![FA-M][Eq-14]

where the function ***diag[•]*** is the operation of setting all elements of the matrix argument to zero except those on the diagonal.

## References
Prince, S.J., 2012. Computer vision: models, learning, and inference. Cambridge University Press.

[Eq-1]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-1.png
[Eq-2]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-2.png
[Eq-3]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-3.png
[Eq-4]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-4.png
[Eq-5]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-5.png
[Eq-6]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-6.png
[Eq-7]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-7.png
[Eq-8]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-8.png
[Eq-9]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-9.png
[Eq-10]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-10.png
[Eq-11]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-11.png
[Eq-12]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-12.png
[Eq-13]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-13.png
[Eq-14]: https://github.com/hgarud/Face_Classification/blob/master/Graphics/Eq-14.png
