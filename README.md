<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Marked in the browser</title>
</head>
<body>
  <div id="content"></div>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    document.getElementById('content').innerHTML =
      marked("$$ \frac{d}{dx} \left(-k\frac{du}{dx}\right) = f \hspace{.6cm} in \hspace{.1cm} ]0, L[, \hspace{.1cm} with \hspace{.1cm} k > 0");
  </script>
</body>
</html>
<!-- ### Model and solve physical problems numerically with respect to thier mathematical model
* Model 1 dimentional elliptic problem -->

<!-- __________________________________________________________________ -->

<!-- ##### 1D elleptic problem : 
$$ \frac{d}{dx} \left(-k\frac{du}{dx}\right) = f \hspace{.6cm} in \hspace{.1cm} ]0, L[, \hspace{.1cm} with \hspace{.1cm} k > 0 \\
u(0) = u_0 \hspace{.5cm} k\frac{du}{dx}(L) = g $$
<head>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
</head> -->