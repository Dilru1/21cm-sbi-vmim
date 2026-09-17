"""Path-B compressor: 3D SE-ResNet on coeval cubes + VMIM head (PlotNeuralNet).

Generates fig_cnn_compressor.tex; compile with pdflatex.
Architecture follows cnn_up.py / thesis Sec. 4.1:
  stem Conv3d 3^3 s1 -> 32@32^3
  RB1 32@32^3 s1 | RB2 64@32^3 s1 | RB3 128@16^3 s2 | RB4 256@8^3 s2 | RB5 512@4^3 s2
  mean+std readout of RB2..RB5 -> concat 1920 -> FC 256 -> FC 128 -> t in R^4
  dashed VMIM head q_phi(theta|t): MLP(64, tanh x3) -> GMM K=2 full cov (training only)
"""
import sys
sys.path.append('../')
from pycore.tikzeng import *

# custom color defs appended to the header
EXTRA_COLORS = r"""
\usepackage{amsmath,amssymb}
\def\SEColor{rgb:red,5;yellow,3;white,4}
\def\PoolStatColor{rgb:green,4;blue,1;white,6}
\def\SummaryColor{rgb:magenta,5;black,2}
\def\HeadColor{rgb:violet,6;white,6}
"""

def raw(s):
    return s

arch = [
    to_head('..'),
    to_cor(),
    raw(EXTRA_COLORS),
    to_begin(),

    # ---------------- input: 3 redshift channels, 32^3 ----------------
    to_Conv("input", s_filer="$32^3$", n_filer=3, offset="(0,0,0)", to="(0,0,0)",
            width=1.5, height=30, depth=30,
            caption="$x\\in\\mathbb{R}^{3\\times32^3}$"),

    raw(r"""
\node[anchor=south,align=center] at ([yshift=30pt]input-north)
    {\scriptsize 3 redshift channels\\[-2pt]\scriptsize $z=8.18,\,10.32,\,12.06$};
"""),

    # ---------------- stem: dense 3^3 conv, stride 1 ----------------
    to_Conv("stem", s_filer="$32^3$", n_filer=32, offset="(1.6,0,0)", to="(input-east)",
            width=2, height=30, depth=30, caption="stem\\\\$3^3$ s1"),
    to_connection("input", "stem"),

    # ---------------- residual + SE trunk ----------------
    to_ConvRes("rb1", s_filer="$32^3$", n_filer=32, offset="(1.5,0,0)", to="(stem-east)",
               width=2.5, height=30, depth=30, opacity=0.4, caption="RB1+SE s1"),
    to_connection("stem", "rb1"),

    to_ConvRes("rb2", s_filer="$32^3$", n_filer=64, offset="(1.5,0,0)", to="(rb1-east)",
               width=3.5, height=30, depth=30, opacity=0.4, caption="RB2+SE s1"),
    to_connection("rb1", "rb2"),

    to_ConvRes("rb3", s_filer="$16^3$", n_filer=128, offset="(1.6,0,0)", to="(rb2-east)",
               width=4.5, height=20, depth=20, opacity=0.4, caption="RB3+SE s2"),
    to_connection("rb2", "rb3"),

    to_ConvRes("rb4", s_filer="$8^3$", n_filer=256, offset="(1.6,0,0)", to="(rb3-east)",
               width=6, height=13, depth=13, opacity=0.4, caption="RB4+SE s2"),
    to_connection("rb3", "rb4"),

    to_ConvRes("rb5", s_filer="$4^3$", n_filer=512, offset="(1.6,0,0)", to="(rb4-east)",
               width=8, height=8, depth=8, opacity=0.4, caption="RB5+SE s2"),
    to_connection("rb4", "rb5"),

    # ---------------- mean+std readout taps under RB2..RB5 ----------------
    raw(r"""
% mean+std pooling taps
\pic[shift={(0,-6.5,0)}] at (rb2-south) {Box={name=p2,caption=,xlabel={{128,}},zlabel=,
    fill=\PoolStatColor,height=4,width=1.2,depth=10,opacity=.7}};
\pic[shift={(0,-6.5,0)}] at (rb3-south) {Box={name=p3,caption=,xlabel={{256,}},zlabel=,
    fill=\PoolStatColor,height=4,width=1.6,depth=10,opacity=.7}};
\pic[shift={(0,-6.5,0)}] at (rb4-south) {Box={name=p4,caption=,xlabel={{512,}},zlabel=,
    fill=\PoolStatColor,height=4,width=2.2,depth=10,opacity=.7}};
\pic[shift={(0,-6.5,0)}] at (rb5-south) {Box={name=p5,caption=,xlabel={{1024,}},zlabel=,
    fill=\PoolStatColor,height=4,width=3,depth=10,opacity=.7}};
\node[anchor=north,text width=4.2cm,align=center] at ([yshift=-14pt]p2-south)
    {\scriptsize per-channel spatial\\[-2pt]\scriptsize $(\mu,\sigma)$ read-out, $2C$};
\draw[connection] (rb2-south) -- node{\midarrow} (p2-north);
\draw[connection] (rb3-south) -- node{\midarrow} (p3-north);
\draw[connection] (rb4-south) -- node{\midarrow} (p4-north);
\draw[connection] (rb5-south) -- node{\midarrow} (p5-north);
"""),

    # ---------------- concat + dense head to t ----------------
    raw(r"""
\pic[shift={(3.5,0,0)}] at (p5-east) {Box={name=cat,caption=concat,xlabel={{1920,}},zlabel=,
    fill=\PoolStatColor,height=4,width=4.2,depth=10,opacity=.9}};
\draw[connection] (p2-east) -- node{\midarrow} (cat-west);
\draw[connection] (p3-east) -- (cat-west);
\draw[connection] (p4-east) -- (cat-west);
\draw[connection] (p5-east) -- node{\midarrow} (cat-west);
\pic[shift={(1.8,0,0)}] at (cat-east) {Box={name=fc1,caption=FC,xlabel={{256,}},zlabel=,
    fill=\FcColor,height=4,width=2.4,depth=10}};
\draw[connection] (cat-east) -- node{\midarrow} (fc1-west);
\pic[shift={(1.2,0,0)}] at (fc1-east) {Box={name=fc2,caption=FC,xlabel={{128,}},zlabel=,
    fill=\FcColor,height=4,width=1.8,depth=10}};
\draw[connection] (fc1-east) -- node{\midarrow} (fc2-west);
\pic[shift={(1.4,0,0)}] at (fc2-east) {Box={name=t,caption=,xlabel={{4,}},zlabel=,
    fill=\SummaryColor,height=4,width=1,depth=6,opacity=.8}};
\draw[connection] (fc2-east) -- node{\midarrow} (t-west);
\node[anchor=north,align=center] at ([yshift=-12pt]t-south)
    {$t=f_\eta(x)\in\mathbb{R}^{4}$\\[-2pt]\scriptsize exported to stage 2};
"""),

    # ---------------- dashed VMIM head (training only) ----------------
    raw(r"""
\pic[shift={(3.2,0,0)}] at (t-east) {Box={name=hd1,caption=,xlabel={{64,}},zlabel=,
    fill=\HeadColor,height=4,width=1.4,depth=8,opacity=.6}};
\pic[shift={(0.9,0,0)}] at (hd1-east) {Box={name=hd2,caption=,xlabel={{64,}},zlabel=,
    fill=\HeadColor,height=4,width=1.4,depth=8,opacity=.6}};
\pic[shift={(0.9,0,0)}] at (hd2-east) {Box={name=hd3,caption=,xlabel={{64,}},zlabel=,
    fill=\HeadColor,height=4,width=1.4,depth=8,opacity=.6}};
\pic[shift={(1.6,0,0)}] at (hd3-east) {Box={name=gmm,caption=,xlabel={{,}},zlabel=,
    fill=\SoftmaxColor,height=6,width=1.2,depth=8,opacity=.7}};
\draw[connection,densely dashed] (t-east) -- node{\midarrow} (hd1-west);
\draw[connection,densely dashed] (hd1-east) -- (hd2-west);
\draw[connection,densely dashed] (hd2-east) -- (hd3-west);
\draw[connection,densely dashed] (hd3-east) -- node{\midarrow} (gmm-west);
\node[anchor=north,align=center] at ([yshift=-10pt]gmm-south)
    {\scriptsize $\{w_k,\mu_k,L_k\}_{k=1}^{2}$};
\node[anchor=south,align=center] at ([yshift=26pt]hd2-north)
    {variational head $q_\varphi(\theta\mid t)$\\[-2pt]\scriptsize MLP $3{\times}64$ (tanh) $\to$ GMM $K{=}2$, full cov.};
\node[anchor=west,align=left] at ([xshift=22pt]gmm-east)
    {$\mathcal{L}_{\mathrm{VMIM}}=-\mathbb{E}\log q_\varphi(\tilde\theta\mid t)$\\[2pt]
     \scriptsize $\sigma_{\min}{=}10^{-2}$ floor, $\theta$-dequantisation\\[-2pt]
     \scriptsize \emph{training only --- head discarded after training}};
% dashed enclosure for the head
\draw[densely dashed, rounded corners=6pt, draw=black!60, thick]
    ([shift={(-14pt,42pt)}]hd1-northwest) rectangle ([shift={(16pt,-34pt)}]gmm-nearsoutheast);
"""),

    to_end()
]

if __name__ == '__main__':
    to_generate(arch, 'fig_cnn_compressor.tex')
    print('wrote fig_cnn_compressor.tex')
