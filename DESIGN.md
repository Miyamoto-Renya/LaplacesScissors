# Laplace’s Scissors · Design

[English](#english) · [한국어](#korean)

<a id="english"></a>

**Computation contracts connecting protected subspaces, finite rotations, and actual GPU validation.**

[Usage](README.md#english) · [Contracts](#computation-contracts) · [Protection](#protected-subspace-and-rotation) · [Candidate planes](#candidate-planes) · [Inequalities](#quadratic-inequalities-for-all-competing-tokens) · [Validation](#runtime-validation) · [Architecture](#components-and-lifetimes)

| Section | Key question |
| --- | --- |
| Computation contracts | Which inputs, outputs, scaling deviations, and divergences are measured? |
| Protected subspace and rotation | Which outputs and matrix properties are preserved? |
| Candidate planes and joint responses | How are edit directions and rotations for multiple targets computed? |
| Quadratic inequalities | How is an angle satisfying every competing-token constraint found? |
| Runtime validation | How are numerical errors and complete generation paths checked? |
| Components and lifetimes | Which module implements each computation? |

---

## Computation contracts

### Answer cases and readout

Inputs are an original model and cases with expected answer strings. Every generated prefix is protected for cases whose original greedy generation satisfies the expected string and EOS conditions. For failed cases, expected tokens are supplied one at a time after the fixed prompt, and the final hidden vector predicting each target token is observed. EOS is also a target token.

Let $h$ be the hidden vector produced by the original decoder, $W$ the output matrix, and $b$ the output bias. Scores are $Wh+b$. A weight edit is represented as a separate readout state $W'=WQ$. The runtime computes $Qh$ and feeds it to the original $W$. When embedding and readout share a stored tensor, embedding calls continue to use the original tensor.

### Saving a proposal as a checkpoint

`export` in [`src/materialize.rs`](src/materialize.rs) saves this state as an independent checkpoint. For an orthonormal basis $U$ and a small rotation $R$, $Q=I+U(R-I)U^T$, so $W'=W+(WU)(R-I)U^T$ is computed by row-block F64 matrix multiplication and stored in FP32. The new output tensor is `lm_head.weight`. Original embedding and decoder tensor bytes are preserved, and `tie_word_embeddings=false` and the output graph are saved. A graph that originally uses `lm_head.weight` in its decoder or embedding requires a separate output name first. Actual generation from the runtime proposal and exported checkpoint checks the effects of storage rounding and operation order. The output directory is finalized at a new path after temporary shape validation and file writing.

### Scaling deviations and recovery

The extension for analyzing stagnation across scales accepts a model series with common evaluation inputs and loss units. For a resource axis $s_k=s_0\rho^k$, hold other training conditions fixed and define the reference decay ratio $q=\rho^{-\alpha}$, $0<q<1$. Run inputs record the intervals used to estimate and evaluate $q$, model revisions, and training resources. For the reference power law $L_k=L_\infty+Aq^k$,

$$
a_k=\frac{1-q^k}{1-q},\qquad
R_k=L_k-a_kL_1+(a_k-1)L_0
$$

is exactly zero. At $k=2$, $R_2=q(L_0-L_1)-(L_1-L_2)$, so a positive value measures a shortfall in expected improvement. Keeping the initial reference points $L_0,L_1$ fixed at larger scales retains the deviation caused by sustained stagnation. Estimate $q$ on the reference interval and use the same value before and after intervention. With per-scale edit effects $b_k=L_k-L_k'$, recovered deviation is

$$
R_k-R_k'=b_k-a_kb_1+(a_k-1)b_0
$$

An equal loss reduction at every scale contributes zero to this expression. [`src/scaling.rs`](src/scaling.rs) fixes the first two reference points and the exponent and applies the same margin, angle, and weight-edit budgets to every checkpoint. It measures expected-answer loss before and after intervention on evaluation prompts separate from repair cases. The loss unit is sequence NLL including EOS per UTF-8 answer byte, averaged across cases. The `scale` manifest supplies resources, fixed conditions, the basis for the exponent, repair cases, evaluation cases, and model revisions.

Arbitrary increasing resource values use the same reference curve. Define $x_k=(s_k/s_0)^{-\alpha}$ and use

$$
a_k=\frac{1-x_k}{1-x_1}
$$

Here $\alpha$ is the fixed manifest exponent. The runtime uses `expm1` to compute differences between nearby reference points.

### Numerical injection and output divergence

The numerical-injection contract constructs a reference vector and search directions from checkpoint embedding rows. It diagnoses output discrimination and residual-path cancellation under the recorded numerical input distribution. Define the two states injected at the same position as

$$
x_\pm=\cos\varepsilon\,x_0\pm\sin\varepsilon\,\|x_0\|v,
\qquad v^Tx_0=0,\quad\|v\|=1
$$

Both states then have equal norms, and injection differences have equal magnitude across search directions. Earlier positions receive fixed mixtures of embedding rows directly. Input row indices, reference state, basis, angle, and injection position are saved together. [`src/injection.rs`](src/injection.rs) selects embedding-row pairs using a seed, constructs the reference state from their sum, and constructs the search basis by orthogonally projecting their differences. Both mixing coefficients are $1/\sqrt2$. The rank of embedding differences determines the actual basis size. `probe` loads weights and execution settings and runs numerical inputs through `Model::inject`. The prompt and expected answer string in `Case` belong to the `diagnose` input path. Both input paths use the same decoder operations.

For output scores $z_\pm$, probabilities $p_\pm=\mathrm{softmax}\,(z_\pm)$, and differences $\delta z=z_+-z_-$ and $\delta p=p_+-p_-$, define divergence as

$$
\mathcal D
=\mathrm{KL}\,(p_+\Vert p_-)+\mathrm{KL}\,(p_-\Vert p_+)
=\delta p^T\delta z.
$$

Both runs use the same softmax support and temperature. Define the logarithmic probability means and matrix

$$
\ell_i=\frac{p_{+,i}-p_{-,i}}{\log p_{+,i}-\log p_{-,i}},\qquad
S=\mathrm{diag}\,(\ell)-\frac{\ell\ell^T}{\mathbf1^T\ell}
$$

In real arithmetic, $\delta p=S\delta z$ and $\mathcal D=\delta z^TS\delta z$ hold exactly. The logarithmic mean of equal probabilities is that probability. The derivation uses $\delta\log p_i=\delta z_i-\delta\log Z$, $\delta p_i=\ell_i\delta\log p_i$, and $\sum_i\delta p_i=0$, giving $\delta\log Z=\ell^T\delta z/(\mathbf1^T\ell)$. A product with $S$ requires a diagonal product and two inner products, so time and storage are linear in vocabulary size. Nearby probabilities use log-softmax differences and the continuous limit of `expm1`.

### Searching for weak discrimination directions

Let $Y=[\delta z_1,\ldots,\delta z_r]$ collect score differences measured along search basis vectors $v_1,\ldots,v_r$, with reference probabilities $p_0$. Form the small matrix

$$
K=Y^T\bigl(\mathrm{diag}\,(p_0)-p_0p_0^T\bigr)Y
$$

Its minimum eigenvector $c$ proposes the next injection direction $v=\sum_i c_iv_i$. This eigenproblem is a local model comparing linear combinations of recorded basis responses. Two actual decoder runs remeasure the selected direction to obtain finite $\mathcal D$. The direction with the smaller measured divergence is selected by comparison with the minimum observed basis divergence. Direction search uses $2r$ executions at the injection position and an $r\times r$ eigenproblem. Layer tracing uses observations from the selected pair of runs.

### Residual branches and coordinate contributions

Write the linear parts of attention and MLP outputs as $B_b a_b$. The final residual difference between two runs is $\delta r_L=\delta x+\sum_b B_b\delta a_b$. Fixed projection biases add the same value in both runs. For final RMSNorm $N(r)=\mathrm{diag}\,(g)r/s$, $s=\sqrt{\|r\|^2/d+\eta}$, use the following finite-difference matrix at the two endpoints:

$$
\bar r=\tfrac12(r_++r_-),\qquad
T_N=\mathrm{diag}\,(g)\left[
\tfrac12(s_+^{-1}+s_-^{-1})I
-\frac{2\bar r\bar r^T}{d\,s_+s_-(s_++s_-)}\right].
$$

It satisfies $\delta N=T_N\delta r$. For LayerNorm, center both inputs with $H=I-\mathbf1\mathbf1^T/d$, construct the matrix above, and multiply it on the right by $H$. Use that product as the effective $T_N$ below. Normalization gain includes the runtime’s gain offset; fixed biases have zero difference. [`src/contrast.rs`](src/contrast.rs) computes products with the transpose of this finite-difference operator. With output matrix $W$, compute $a=T_N^TW^T\delta p$ once to obtain

$$
\boxed{\mathcal D=a^T\delta x+\sum_b\sum_{i,j}
 a_i(B_b)_{ij}(\delta a_b)_j.}
$$

This identity decomposes the fixed endpoint pair’s output divergence into signed contributions by branch and weight coordinate. First, measured branch-output differences yield $a^T\delta(B_ba_b)$ to select a branch; only the selected matrix is expanded into coordinates. Storage for a full weight Jacobian is replaced by vector observations and a traversal of the selected matrix. [`src/injection.rs`](src/injection.rs) traverses the matrix with the largest negative branch contribution, collects coordinates in a bounded heap, and computes attention-head totals. It records the numerical residual between the coordinate-contribution sum and measured GPU divergence. Coordinate intervention effects are measured by applying separate sparse state to projection calls of the same tensor and rerunning from the fixed numerical prefix. Signed contributions and finite intervention effects retain their respective computational meanings in reports.

### Protected rotations that increase divergence

The edit direction is chosen to increase divergence between the selected final hidden states $h_\pm$. The protected subspace $C$ contains original protected vectors, numerical control states, and $\bar h=(h_++h_-)/2$. Let $P_C$ project onto its orthogonal complement and let $F_\pm=\mathrm{diag}\,(p_\pm)-p_\pm p_\pm^T$. Define

$$
a_h=P_C(h_+-h_-),\qquad
b_h=P_CW^T\left[\delta p+\tfrac12(F_++F_-)\delta z\right],
$$

$$
u=\frac{a_h}{\|a_h\|},\qquad
b_\perp=b_h-u(u^Tb_h),\qquad v=\frac{b_\perp}{\|b_\perp\|},
$$

$$
Q(\theta)=I+(\cos\theta-1)(uu^T+vv^T)
+\sin\theta(vu^T-uv^T).
$$

When $a_h$ and $b_\perp$ have positive norms, $\mathcal D'(0)=\|a_h\|\|b_\perp\|>0$. This is the analytic angular derivative of divergence for the two fixed hidden states. Finite rotations are measured through actual readout. In real arithmetic, the rotation preserves $QC=C$, $Q^TQ=I$, $W'W'^T=WW^T$, and the midpoint of the two output-score vectors. Angle and weight-edit budgets follow the existing readout contract. Fixing the midpoint removes the freedom to increase divergence by moving the center state.

The implementation expresses this direction using the existing half-angle sign convention. When the first-order direction degenerates, it projects up to 32 row differences, referenced to a high-probability output row, onto the protected subspace’s complement to construct finite candidates. Acceptance uses measured divergence improvement and protection conditions. The weight Frobenius budget further restricts the angle bound. Actual F32 rotation and readout are measured at 0.04, 0.2, and 1 times that bound. Protected states include the input reference state, the two control-injection states orthogonal to the selected direction, and additional protected vectors from the input file. The actual output-midpoint change is measured as well. Version 3 proposals store the divergence objective, numerical-input SHA256, and weight and execution-configuration hashes. `probe --inputs ... --proposal ...` reruns the full decoder on saved inputs.

Sources of inspiration include [caustics of reflected billiard rays](https://arxiv.org/abs/2112.07852), [logarithmic mean temperature differences in heat exchangers](https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node131.html), and [entropy flow in finite Markov chains](https://www.janmaas.org/papers/discrete.pdf). The injection distribution, finite-difference softmax matrix, residual decomposition, and divergence derivative under protected rotation above are derived from this project’s operation contracts.

## Protected subspace and rotation

Collect protected hidden vectors as columns of $C$, and choose orthonormal unit vectors $u,v$ in the orthogonal complement of its column space.

$$
u^TC=v^TC=0,\qquad P=uu^T+vv^T,\qquad J=uv^T-vu^T.
$$

Use the half-angle coordinate $t=\tan(\theta/2)$:

$$
Q(t)=I-\frac{2t^2}{1+t^2}P+\frac{2t}{1+t^2}J.
$$

From $J^T=-J$, $J^2=-P$, and $P^2=P$, we obtain $Q^TQ=I$. The following identities therefore hold in real arithmetic:

$$
W'C=WC,\qquad W'W'^T=WW^T,\qquad
\mathrm{rank}\,(W'-W)\le2.
$$

Full-vocabulary scores at protected prefixes and inner products between output rows are preserved. The rotation acts within the plane spanned by $u,v$. [`src/rotation.rs`](src/rotation.rs) builds the protected-space basis with two orthogonal-projection passes and a relative rank threshold of $10^{-10}$.

## Candidate planes

### Plane selection

Let $d=w_y-w_j$ be the output-row difference between target token $y$ and the current highest-scoring competitor $j$, $m$ the observed margin, and $\tau$ the required margin. Use the shortfall $\lambda=\max(\tau-m,0)$ as a weight. With $\bar d,\bar h$ projected onto the protected subspace’s orthogonal complement, form the skew-symmetric operator

$$
K=\sum_i\lambda_i(\bar d_i\bar h_i^T-\bar h_i\bar d_i^T).
$$

In the span of observed vectors, compute an eigenvector $u$ of $K^TK$ and set $v=-Ku/\|Ku\|$ to form a rotation plane. The plane associated with the largest eigenvalue maximizes the first-order increase in the weighted margin sum among normalized plane rotations. Further eigenvector planes and individual $(\bar d_i,\bar h_i)$ planes supply additional candidates; each distinct plane is checked once. `max_planes` sets the candidate count.

This stage selects the edit plane. Each plane’s angle is computed from the finite-rotation conditions below.

### Linear response in observation space

An extension that distributes requested improvements in scaling deviation across multiple observations operates in observation space. Let $f_i$ be observations to increase and $p_j$ observations to protect. Stack their derivatives in fixed edit coordinates as rows of $J,B$. Use $f_i=-L_i$ for losses and margins for token constraints. The protection projection and response matrix are

$$
\Pi=I-B^T(BB^T)^\dagger B,\qquad
K=J\Pi J^T.
$$

For the linear constraints $J\Delta w=r,\ B\Delta w=0$ with $r\in\mathrm{range}\,(K)$,

$$
\lambda=K^\dagger r,\qquad
\Delta w=\Pi J^T\lambda,\qquad
\|\Delta w\|^2=r^TK^\dagger r
$$

gives the minimum Euclidean edit in those coordinates. Constructing the full observation Gram matrix, including protection terms, and computing $K=G_{ff}-G_{fp}G_{pp}^\dagger G_{pf}$ reduces the storage required for the global projection $\Pi$. The vector $\lambda$ coordinates responses among observations that must be satisfied together.

For affine weights $W_\ell$, let $x_{\ell i t}$ be their inputs and $d_{\ell i t}$ the derivatives propagated backward from an observation. Then

$$
\nabla_{W_\ell}f_i=\sum_t d_{\ell i t}x_{\ell i t}^T,
$$

$$
G_{ij}^{(\ell)}
=\sum_{t,u}
(d_{\ell i t}^Td_{\ell j u})
(x_{\ell i t}^Tx_{\ell j u}).
$$

Summing these terms across layers produces the same Gram matrix as an explicit weight Jacobian. A tensor used at multiple positions includes all its uses in the sum. Other weight operations, such as normalization and embeddings, contribute inner-product terms from their derivative contracts. This expression specifies the computation contract for connecting derivatives across all layers. The implemented response calculation uses the readout skew-symmetric coordinates below.

### Joint readout rotations

The same response expression applies directly to the current readout rotation. Using $\bar d_i,\bar h_i$ projected onto the protected subspace’s orthogonal complement, form $A_i=\bar d_i\bar h_i^T-\bar h_i\bar d_i^T$. With the skew-symmetric inner product $\langle A,B\rangle=\tfrac12\mathrm{tr}\,(A^TB)$,

$$
K_{ij}
=(\bar d_i^T\bar d_j)(\bar h_i^T\bar h_j)
-(\bar d_i^T\bar h_j)(\bar h_i^T\bar d_j),\qquad
\Omega=\sum_i(K^\dagger r)_iA_i.
$$

$\Omega$ is the first-order edit with minimum angular norm that produces the requested margin changes. Its image lies in the span of observed vectors, with rank at most $2m$. Given an orthonormal basis $U$ of that space and $\omega=U^T\Omega U$, compute the finite edit as

$$
Q_U(t)=(I-t\omega/2)^{-1}(I+t\omega/2),\qquad
Q(t)=I+U(Q_U(t)-I)U^T
$$

Orthogonality and the protected subspace are preserved in real arithmetic. [`src/response.rs`](src/response.rs) computes the joint direction using the observation Gram pseudoinverse. After plane candidates have been checked, [`src/diagnosis.rs`](src/diagnosis.rs) applies this direction and selects a step size that improves margins in compressed GPU outputs. Each step enforces bounds on the accumulated rotation’s largest principal angle and weight-edit norm. Vectors from originally successful paths and target prefixes already satisfying the required margin are fixed by excluding them from the basis. Newly competing tokens enter the next step’s response calculation. Candidate records store linear-response residuals and final measurements from the original readout separately.

### Compressing outputs and costs

Joint search reduces computation using a fixed basis $U\in\mathbb R^{d\times r}$ that respects the protection constraints, a local rotation $R\in SO(r)$, and observation coordinates $x=U^Th$. Compute $F=WU$, $G=F^TF$, and original scores $z_0=Wh+b$ once. Then

$$
z_R=z_0+F(R-I)x,\qquad
\mathcal E(R)=\|W'-W\|_F^2
=\mathrm{tr}\,((R-I)^TG(R-I))
=2\mathrm{tr}\,G-2\mathrm{tr}\,(GR).
$$

These identities require $U^TU=I$ and $R^TR=I$. While the basis and original weights remain fixed, $F,G,z_0,x$ are shared. Initial Gram construction costs $O(Vr^2)$; subsequent stable quadratic-form costs take $O(r^3)$, and trace costs take $O(r^2)$. The matrix width for output evaluation falls from $d$ to $r$. For small rotations, the quadratic form directly computes the small positive change. Measurements from the original readout check GPU F32 multiplication-order and storage-rounding errors.

### Reducing the Cayley system

The joint direction $\Omega=\sum_i\lambda_i(d_ix_i^T-x_id_i^T)$ factors as $\Omega=SJS^T$, where $a_i=\sqrt{|\lambda_i|}d_i$, $b_i=\mathrm{sgn}\,(\lambda_i)\sqrt{|\lambda_i|}x_i$, $S=[a_1,\ldots,a_m,b_1,\ldots,b_m]$, and $J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}$. Let $K=S^TS$. Then

$$
T_t=t(I-tJK/2)^{-1}J,\qquad
\mathrm{Cayley}\,(t\Omega)=I+ST_tS^T,\qquad
R_t=R+RST_tS^T.
$$

This exact reduction follows from $(I-aSJS^T)S=S(I-aJS^TS)$. The determinants also satisfy $\det(I-aSJS^T)=\det(I-aJS^TS)$, and the conjugate eigenvalue pairs of skew-symmetric $\Omega$ make this value positive for real step sizes. The expression also holds when the columns of $S$ are linearly dependent. Candidates with $2m<r$ can share the smaller $2m$-dimensional linear system. Prepare $D=R-I$, $P=S^TD^TGRS$, and $N=S^TR^TGRS$ before step-size search to obtain

$$
\mathcal E(R_t)=\mathcal E(R)+2\mathrm{tr}\,(PT_t)
+\mathrm{tr}\,(T_t^TNT_tK).
$$

At the initial $R=I$, the final positive quadratic term directly gives the edit cost. Per-candidate cost computation uses matrices of size $2m$. For $2m<r$, `ResponsePath` constructs $S,J,K,P,N$ and the reference cost once per step and shares them across five step sizes. For $2m\ge r$, it solves the $r$-dimensional Cayley system. When the incremental cost falls within a rounding bound computed from term magnitudes, it uses the stable quadratic form in `WeightMetric`. The accumulated rotation’s largest principal angle is checked under the existing angle contract.

### Minimum-cost completion preserving observations

A further reduction in observation space uses isometric extensions of rotations that produce the same observations. Let $E\in\mathbb R^{r\times k}$ be a column-space basis of $H$, which collects all coordinates subject to change, and let $Y=RE$ be its image. Output conditions depend on $Y$. We have $Y^TY=I$, and the observation movement has $rk-k(k+1)/2$ degrees of freedom. For $0<k<r$, complete $[E,E_\perp]$ and $[Y,Y_\perp]$ to orientation-preserving orthogonal bases. Every rotation with the same observations then has the form

$$
R_B=YE^T+Y_\perp B E_\perp^T,\qquad B\in SO(r-k)
$$

Let $M=E_\perp^TGY_\perp=L\Sigma V^T$, $\sigma_1\ge\cdots\ge\sigma_{r-k}$, and $\delta=\det(VL^T)$. Then

$$
B_*=V\mathrm{diag}\,(1,\ldots,1,\delta)L^T,
$$

$$
\min_B\mathcal E(R_B)
=2\mathrm{tr}\,G-2\mathrm{tr}\,(E^TGY)
-2\left(\sum_{i<r-k}\sigma_i+\delta\sigma_{r-k}\right).
$$

The proof splits $\mathrm{tr}\,(GR_B)=\mathrm{tr}\,(E^TGY)+\mathrm{tr}\,(MB)$ and maximizes the singular-value trace subject to the orientation constraint. When the completed $R_{B_*}$ satisfies the existing angle bound, it preserves observed scores with minimum weight-edit cost. Orthogonality to $U$ maintains protected states. At the start of search, `ObservedSpace` fixes the observation column space by selecting directions whose singular values exceed $\max(10^{-12},10^{-9}\sigma_{\max})$. At observation rank 0, the identity rotation attains the minimum; at rank $r$, the given rotation does.

[`src/response.rs`](src/response.rs) computes cached costs, reduced Cayley systems, and minimum-cost completion. `CompactReadout` in [`src/diagnosis.rs`](src/diagnosis.rs) loads $F$ and reference scores once in GPU F32. It computes $(R-I)x$ in F64, converts it to F32, and evaluates it through the shared `Gpu::linear`. Both the final search rotation and the completed candidate are measured through the original GPU readout at every observed prefix. Completion is accepted when it preserves angle and cost bounds, all prefix argmax choices, already satisfied target margins, and protected-output tolerance, with scaled logit differences between the two candidates within tolerance. This check covers numerical rank truncation and F32 operation order. When the original candidate meets all acceptance conditions, the completed candidate must meet those same conditions. `completion` records costs before and after completion, observation rank, measured output differences, and whether completion was applied. `computation` records basis-output, compressed-output, and final-validation call counts; cost-construction counts; trials by rotation solver; and algebra timings for joint preparation, search, and completion.

Mathematical sources of inspiration include [Witt’s isometric extension for quadratic forms](https://swc-math.github.io/aws/2009/09HankeNotes.pdf) and [operator-space reduction in the Ihara zeta function](https://www.math.lsu.edu/system/files/jwh2002b.pdf). The cost, response-reduction, and minimum-edit-completion formulas above are derived from the project’s readout contract.

Further sources of inspiration include [effective resistance and response matrices in electrical networks](https://www.cs.yale.edu/homes/spielman/462/2007/lect9-07.pdf) and [annihilating filters for impulse-signal reconstruction](https://bigwww.epfl.ch/publications/vetterli0201.html). Scaling deviations, protected responses, and rotation Gram matrices are derived directly from the computation contracts above.

## Quadratic inequalities for all competing tokens

Let $m$ be the original score difference between the target and a competing token at a fixed prefix. It already includes the bias. For each competitor,

$$
\alpha=d^TJh,\qquad \beta=d^TPh,
$$

$$
m(t)=m+\frac{2\alpha t-2\beta t^2}{1+t^2}.
$$

Since $1+t^2>0$, $m(t)\ge\tau$ is exactly equivalent to

$$
(m-\tau-2\beta)t^2+2\alpha t+(m-\tau)\ge0.
$$

[`src/diagnosis.rs`](src/diagnosis.rs) computes the two output directions once per plane and constructs all coefficients from each prefix’s two projections. [`src/interval.rs`](src/interval.rs) merges the open intervals forbidden by each constraint, maintaining a common solution for all target tokens against every vocabulary competitor.

The weight change is

$$
\|WQ-W\|_F^2=
\frac{4t^2}{1+t^2}\bigl(\|Wu\|^2+\|Wv\|^2\bigr).
$$

For a fixed plane, this grows with $|t|$. The feasible $t$ closest to zero therefore minimizes weight change within that plane. Candidates across planes are validated in order of this change. The minimum is taken over the allowed rotation angles of each selected plane.

### Forbidden intervals and the feasible set

The same solution can be computed through forbidden intervals. Define $c_i=m_i-\tau$, $a_i=c_i-2\beta_i$, and $b_i=2\alpha_i$. The open intervals where $q_i(t)=a_it^2+b_it+c_i$ is negative form that constraint’s forbidden region. In angular coordinates,

$$
m_i(\theta)-\tau=c_i-\beta_i+\beta_i\cos\theta+\alpha_i\sin\theta
$$

so each constraint defines a forbidden arc of a circle. In half-angle coordinates, it becomes at most two open intervals. If $U$ is the union of forbidden intervals and $[-T,T]$ is the half-angle range, the feasible set is $F=[-T,T]\setminus U$. When zero belongs to $U$, the minimum-edit solution is the in-range endpoint with smallest absolute value among the two endpoints of the connected component containing zero. Two open intervals that touch at an endpoint leave that point feasible.

Tracking the remaining feasible set’s endpoints $L=\min F$ and $R=\max F$ reduces the constraints further. If $q_i\ge0$ throughout $[L,R]$, the constraint also holds for every subsequent feasible set. This check requires only the endpoints and any vertex inside the interval. For $a_i>0$ and $-b_i/(2a_i)\in[L,R]$, the vertex condition is $4a_ic_i-b_i^2\ge0$. Otherwise, endpoint signs determine the result. Roots are computed and added to the union for constraints that actually cut the interval.

Merging overlapping forbidden intervals in a balanced search tree gives O(N + K log(K+1)) time and O(K) storage for N constraints and K generated forbidden intervals. Each interval removed by merging is removed once. [`src/interval.rs`](src/interval.rs) uses this representation and endpoint/vertex screening. Coefficients are normalized by powers of two, preserving singleton solutions between touching open intervals and symmetric minimum solutions. Reports record square-root and screening counts.

Sources of inspiration include [circular projection and shadows in sundials](https://www.dezonnewijzerkring.nl/downloads/JdR-EquatorProj.pdf) and [covering a circle with random arcs](https://statistics.stanford.edu/technical-reports/covering-circle-random-arcs-random-sizes). The forbidden intervals and cost formula above follow directly from transforming the program’s margin expression.

## Runtime validation

### Precision and acceptance conditions

Planes, bases, and inequalities use F64; decoder and rotation kernels use F32. Original F32/F16/BF16 weights are read in their stored form. Inequality targets include a numerical allowance of $8\epsilon\max(1,\max|\mathrm{logit}|)$, where $\epsilon$ is the user’s validation tolerance.

For each candidate, the GPU measures:

1. The actual minimum margin between the target token and all competitors at every target prefix.
2. Full-output changes at protected prefixes, both absolute and divided by $1+|\mathrm{original\ score}|$.
3. The complete answer and EOS termination implied by argmax choices along observed prefixes.
4. Preservation of original generated token sequences for protected cases and agreement with target sequences for failed cases.

A proposal is accepted when measured margins meet the requirement, protection errors stay within tolerance, and tokens along the observed path agree through EOS. With a fixed decoder, selecting each target token with a positive margin and selecting the final EOS establishes the complete greedy target answer by induction over prefixes. Full-model regression tests compare this decision with generation after reapplying the proposal.

### Prefix validation and full generation

For the hidden vector $h_i$ of a fixed prefix $p_i$, readout rotation satisfies $h_Q(p_i)=h_0(p_i)$. Applying actual GPU rotation and original readout to the observed $h_i$ therefore gives the prefix’s actual edited output $z_i(Q)=WQh_i+b$. Selecting the first target token makes the next prefix equal to the observed path; repeating this relation through EOS verifies the full generation. Protected cases use the same induction by checking score changes and selected tokens at every prefix of the original generation path.

[`src/diagnosis.rs`](src/diagnosis.rs) keeps each observation’s prompt, token order, and hidden vector together. Candidate validation executes actual GPU rotation, readout, and argmax on those vectors. When all tokens match, it verifies the entire path; at the first mismatch, it records the generation prefix reached at that point. The validation function takes a shared model reference, keeping decoder state fixed. Plane edits use the existing F32 rotation operation order. Joint edits use basis projection, a small matrix product, and basis reconstruction. `generate` reapplies a saved proposal during full-decoder execution.

### Attribution along an intervention path

Weight-level attribution is measured along a specified intervention path $\theta(t)$. Define each coordinate’s contribution to scaling deviation $R_k$ as

$$
C_a=-\int_0^1
\frac{\partial R_k(\theta(t))}{\partial\theta_a}
\dot\theta_a(t)\,dt
$$

The chain rule gives $\sum_a C_a=R_k(\theta(0))-R_k(\theta(1))$. All edited coordinates are included, with contributions that increase deviation retained as negative values. The path and weight groupings are part of the run conditions. Along a straight path, $\dot\theta=\Delta\theta$, and the loss derivative at a fixed readout prefix is $(\mathrm{softmax}\,(z)-e_y)h^T$. A straight path between endpoints satisfying $\Delta WC=0$ preserves protected scores. Orthogonality and output-row Gram preservation apply to the rotation path and final rotation.

[`src/attribution.rs`](src/attribution.rs) integrates the softmax derivative along the straight readout path and records the error between the contribution sum and measured loss reduction. Integration stops when the sum residual reaches tolerance; at the maximum subdivision count, it still records the measured residual and convergence status. Output-row contributions use actual F32 endpoint scores. Coordinate contributions within a row are reconstructed from F64 weight changes, with their difference from the row contribution recorded. The loss after reverting high-contribution rows to their original values is also computed. This row reversion specifies the finite intervention used for attribution. The accepted proposal contains the complete weight state of the joint rotation.

Scaling reports multiply each checkpoint’s contributions by its scale-contrast coefficient. Row and column coordinates are recorded with each checkpoint’s tensor names. Contributions from displayed rows and remaining rows are summed and compared with total recovered deviation. The same coefficients measure the increase in deviation caused by row reversion. Protection in the global response expression is a first-order derivative condition; the implemented readout rotation is a finite transformation that fixes the protected subspace.

Saved proposals contain rotation vectors, half-angle coordinates, readout connections, model revisions, and hashes of all input files. Original checkpoints are loaded read-only, with edit state owned separately at runtime.

## Components and lifetimes

[`src/hub.rs`](src/hub.rs) resolves revisions, fetches original files, and computes hashes. [`src/graph.rs`](src/graph.rs) establishes decoder operation contracts from configuration and tensor shapes. [`src/text.rs`](src/text.rs) prepares the original tokenizer and chat template, fixed prompts, and answer continuations. [`src/probes.rs`](src/probes.rs) provides user inputs and symbolic cases with computable answers.

[`src/model.rs`](src/model.rs) runs the shared decoder for token and numerical inputs, KV caching, observations of residual branches and final hidden vectors, separate projection interventions, and readout state. [`src/gpu.rs`](src/gpu.rs) manages tensors, kernel arguments, and compilation metadata. [`src/device.rs`](src/device.rs) manages CUDA/HIP memory, context, and module lifetimes. [`kernels/ops.py`](kernels/ops.py) and [`kernels/compile.py`](kernels/compile.py) implement Triton operations and the compilation bridge. Kernel-source and execution-target content hashes identify cache entries.

[`src/rotation.rs`](src/rotation.rs) computes protected subspaces and candidate planes; [`src/interval.rs`](src/interval.rs) solves forbidden intervals; [`src/response.rs`](src/response.rs) computes joint responses and small rotations. [`src/diagnosis.rs`](src/diagnosis.rs) observes prefixes, computes candidates, validates GPU readout, and saves proposals. [`src/contrast.rs`](src/contrast.rs) computes finite probability and normalization differences and divergence derivatives. [`src/injection.rs`](src/injection.rs) handles numerical-input construction, direction search, cancellation coordinates, finite interventions, protected rotations, and replay. [`src/attribution.rs`](src/attribution.rs) computes weight contributions and row-reversion effects. [`src/scaling.rs`](src/scaling.rs) computes deviations and recovery across checkpoint series. [`src/main.rs`](src/main.rs) exposes model acquisition, input generation, diagnosis, scaling analysis, and proposal reapplication. Run results and compilation artifacts are stored outside the project.

---

<a id="korean"></a>

**한국어** · [English](#english)

**보호 공간, 유한 회전, 실제 GPU 검증을 연결하는 계산 계약.**

[사용법](README.md#korean) · [계산 계약](#계산-계약) · [보호 공간](#보호-공간과-회전) · [후보 평면](#후보-평면) · [이차부등식](#모든-경쟁-token에-대한-이차부등식) · [검증](#실제-실행의-검증) · [구성](#구성과-수명)

| 읽을 내용 | 핵심 질문 |
| --- | --- |
| 계산 계약 | 어떤 입력과 출력, 규모 편차와 구별도를 측정하는가? |
| 보호 공간과 회전 | 어떤 출력과 행렬 성질을 보존하는가? |
| 후보 평면과 공동 응답 | 수정 방향과 여러 목표의 회전을 어떻게 구하는가? |
| 이차부등식 | 전체 경쟁 token의 조건을 만족하는 각도를 어떻게 구하는가? |
| 실제 실행의 검증 | 수치 오차와 전체 생성 경로를 어떻게 확인하는가? |
| 구성과 수명 | 각 계산을 어느 모듈이 담당하는가? |

---

## 계산 계약

### 정답 사례와 readout

입력은 원본 모델과 정답 문자열을 가진 사례다. 원본 greedy 생성이 정답 문자열 및 EOS 조건을 만족한 사례의 모든 생성 접두사를 보호한다. 실패 사례에서는 고정 prompt 뒤에 정답 token을 하나씩 공급해 각 목표 token을 예측하는 최종 내부 벡터를 관측한다. EOS도 목표 token으로 포함한다.

원본 decoder가 만든 내부 벡터를 $h$, 출력 행렬을 $W$, 출력 bias를 $b$라 두면 점수는 $Wh+b$다. 가중치 수정은 별도 readout 상태 $W'=WQ$로 표현한다. 실행기는 $Qh$를 계산해 원본 $W$에 입력한다. embedding과 readout이 같은 저장 tensor를 참조하는 경우에도 embedding 호출은 원본 tensor를 사용한다.

### 수정안의 checkpoint 저장

[`src/materialize.rs`](src/materialize.rs)의 `export`는 이 상태를 독립 checkpoint로 저장한다. 직교 기저 $U$와 작은 회전 $R$에 대해 $Q=I+U(R-I)U^T$이므로 $W'=W+(WU)(R-I)U^T$를 행 블록별 F64 행렬 곱으로 계산하고 FP32로 기록한다. 새로운 출력 tensor는 `lm_head.weight`이며, 원본 embedding과 decoder tensor의 바이트를 보존하고 `tie_word_embeddings=false`와 출력 그래프를 저장한다. 원본에서 `lm_head.weight`를 decoder나 embedding에 사용하는 그래프는 독립 출력 이름을 먼저 정해야 한다. 저장 반올림과 연산 순서의 영향은 실행 상태 수정안과 새 checkpoint의 실제 생성 결과로 확인한다. 출력 폴더는 임시 형상 검사와 파일 작성을 마친 뒤 새 경로로 확정한다.

### 규모 편차와 회복량

규모에 따른 정체를 분석하는 확장 계약은 같은 평가 입력과 손실 단위를 사용하는 모델 계열을 받는다. 자원 축 $s_k=s_0\rho^k$에서 비교할 때 다른 학습 조건을 고정하고, 기준 구간의 감소율 $q=\rho^{-\alpha}$, $0<q<1$를 정한다. $q$의 추정 구간과 평가 구간, 모델 revision과 학습 자원을 실행 입력에 기록한다. 기준 거듭제곱 $L_k=L_\infty+Aq^k$에 대해

$$
a_k=\frac{1-q^k}{1-q},\qquad
R_k=L_k-a_kL_1+(a_k-1)L_0
$$

는 정확히 0이다. $k=2$에서는 $R_2=q(L_0-L_1)-(L_1-L_2)$이므로 양의 값이 기대 개선량의 부족분을 나타낸다. 고정된 초기 기준점 $L_0,L_1$을 이후 규모에도 사용하면 지속되는 정체의 편차가 남는다. 기준 구간에서 $q$를 추정한 뒤 개입 전후에 같은 값을 사용한다. 규모별 수정 효과 $b_k=L_k-L_k'$의 편차 회복량은

$$
R_k-R_k'=b_k-a_kb_1+(a_k-1)b_0
$$

이다. 모든 규모의 손실을 똑같이 낮추는 효과는 이 식에서 0이 된다. [`src/scaling.rs`](src/scaling.rs)는 처음 두 기준점과 지수를 고정하고 모든 checkpoint에 같은 margin·각도·가중치 수정 예산을 적용한다. 수정 사례와 분리한 평가 prompt에서 개입 전후의 정답 문자열 손실을 측정한다. 손실 단위는 EOS를 포함한 sequence NLL의 정답 UTF-8 바이트당 값의 사례 평균이다. `scale` 명령의 manifest는 자원량·고정 조건·지수의 근거·수정 사례·평가 사례·모델 revision을 받는다.

임의의 증가하는 자원량에도 같은 기준 곡선을 사용한다. $x_k=(s_k/s_0)^{-\alpha}$라 두고

$$
a_k=\frac{1-x_k}{1-x_1}
$$

를 적용한다. $\alpha$는 manifest의 고정 지수다. 실행기는 $\mathrm{expm1}$으로 가까운 기준점의 차를 계산한다.

### 수치 주입과 출력 구별도

수치 주입 계약은 checkpoint의 embedding 행에서 기준 벡터와 탐색 방향을 만든다. 진단 대상은 기록한 수치 입력 분포에서의 출력 구별도와 잔차 경로의 상쇄다. 같은 자리에 주입하는 두 상태를

$$
x_\pm=\cos\varepsilon\,x_0\pm\sin\varepsilon\,\|x_0\|v,
\qquad v^Tx_0=0,\quad\|v\|=1
$$

로 정하면 두 상태의 길이가 같고, 모든 탐색 방향의 주입 차이도 같다. 앞선 위치에는 embedding 행의 고정 혼합을 직접 공급한다. 입력 행 인덱스·기준 상태·기저·각도·주입 위치를 함께 저장한다. [`src/injection.rs`](src/injection.rs)는 seed로 embedding 행 쌍을 고르고 합으로 기준 상태를, 차이의 직교 투영으로 탐색 기저를 만든다. 혼합 계수는 각각 $1/\sqrt2$다. 실제 기저 크기는 embedding 차이의 rank에 맞춘다. `probe` 명령은 가중치·실행 설정을 적재하고 `Model::inject`로 수치 입력을 실행한다. `Case`의 prompt와 정답 문자열은 `diagnose` 입력 경로다. 두 입력 경로는 같은 decoder 연산을 사용한다.

출력 점수 $z_\pm$, 확률 $p_\pm=\mathrm{softmax}\,(z_\pm)$, 차이 $\delta z=z_+-z_-$, $\delta p=p_+-p_-$에 대해 구별도를 다음처럼 정한다.

$$
\mathcal D
=\mathrm{KL}\,(p_+\Vert p_-)+\mathrm{KL}\,(p_-\Vert p_+)
=\delta p^T\delta z.
$$

두 실행은 같은 softmax support와 온도를 사용한다. 확률의 로그평균을

$$
\ell_i=\frac{p_{+,i}-p_{-,i}}{\log p_{+,i}-\log p_{-,i}},\qquad
S=\mathrm{diag}\,(\ell)-\frac{\ell\ell^T}{\mathbf1^T\ell}
$$

로 정의하면 실수 연산에서 $\delta p=S\delta z$, $\mathcal D=\delta z^TS\delta z$가 정확히 성립한다. 같은 확률의 로그평균은 그 확률값이다. 유도는 $\delta\log p_i=\delta z_i-\delta\log Z$, $\delta p_i=\ell_i\delta\log p_i$, $\sum_i\delta p_i=0$를 사용한다. 따라서 $\delta\log Z=\ell^T\delta z/(\mathbf1^T\ell)$다. $S$의 벡터 곱은 대각 곱과 두 내적으로 계산하며 시간·저장량은 vocabulary 크기에 선형이다. 가까운 확률은 log-softmax 차이와 `expm1`의 연속 극한으로 계산한다.

### 약한 구별 방향 탐색

탐색 기저 $v_1,\ldots,v_r$에서 실행한 점수 차이를 $Y=[\delta z_1,\ldots,\delta z_r]$라 하고 기준 확률을 $p_0$라 둔다. 작은 행렬

$$
K=Y^T\bigl(\mathrm{diag}\,(p_0)-p_0p_0^T\bigr)Y
$$

의 최소 고유벡터 $c$로 다음 주입 방향 $v=\sum_i c_iv_i$를 정한다. 이 고유문제는 기록한 기저 응답의 선형 결합을 비교하는 국소 모형이다. 선택한 방향은 두 실제 decoder 실행으로 다시 측정해 유한한 $\mathcal D$를 얻는다. 기저에서 관측한 최소 구별도와 비교해 실제 값이 작은 방향을 채택한다. 방향 탐색에는 $2r$개의 해당 위치 실행과 $r\times r$ 고유문제를 사용한다. 선택한 쌍의 층별 추적에는 두 실행의 관측값을 사용한다.

### 잔차 분기와 좌표 기여

층별 상쇄를 추적할 때 attention 출력과 MLP 출력의 선형 부분을 $B_b a_b$로 모으면 두 실행의 최종 residual 차이는 $\delta r_L=\delta x+\sum_b B_b\delta a_b$다. 고정 projection bias는 두 실행에서 같은 값을 더한다. 최종 RMSNorm $N(r)=\mathrm{diag}\,(g)r/s$, $s=\sqrt{\|r\|^2/d+\eta}$의 두 끝점에는 다음 유한 차이 행렬을 사용한다.

$$
\bar r=\tfrac12(r_++r_-),\qquad
T_N=\mathrm{diag}\,(g)\left[
\tfrac12(s_+^{-1}+s_-^{-1})I
-\frac{2\bar r\bar r^T}{d\,s_+s_-(s_++s_-)}\right].
$$

$\delta N=T_N\delta r$가 성립한다. LayerNorm은 $H=I-\mathbf1\mathbf1^T/d$로 두 입력을 중심화해 위 행렬을 구한 뒤 오른쪽에 $H$를 곱하고, 그 곱을 아래 식의 유효 $T_N$으로 사용한다. 정규화 gain에는 실행기의 gain offset을 포함하며 고정 bias의 차이는 0이다. [`src/contrast.rs`](src/contrast.rs)는 이 유한 차이 전치를 벡터로 계산한다. 출력 행렬을 $W$라 하고 $a=T_N^TW^T\delta p$를 한 번 계산하면

$$
\boxed{\mathcal D=a^T\delta x+\sum_b\sum_{i,j}
 a_i(B_b)_{ij}(\delta a_b)_j.}
$$

이 식은 고정한 두 끝점의 출력 구별도를 분기·가중치 좌표별 부호 기여로 분해한다. 먼저 실제 분기 출력 차이로 $a^T\delta(B_ba_b)$를 계산해 분기를 고르고, 선택한 행렬만 좌표 단위로 펼친다. 전체 가중치 Jacobian을 구성하는 저장량을 벡터 관측과 선택한 행렬의 순회 비용으로 바꾼다. [`src/injection.rs`](src/injection.rs)는 음의 분기 기여가 가장 큰 행렬을 순회하며 제한된 heap에 좌표를 모으고 attention 헤드별 합도 계산한다. 좌표 기여의 합과 실제 GPU 구별도 사이의 수치 잔차를 기록한다. 가중치 좌표의 개입 효과는 별도 sparse 상태를 같은 tensor의 projection 호출에 적용하고 고정한 수치 접두사부터 다시 실행해 측정한다. 부호 기여와 유한 개입 효과는 각각의 계산 의미로 보고한다.

### 구별도를 높이는 보호 회전

수정 방향은 선택한 두 최종 hidden $h_\pm$의 구별도를 증가시키도록 계산한다. 보호 공간 $C$에는 원본 보호 벡터, 수치 대조 상태와 $\bar h=(h_++h_-)/2$를 넣는다. 그 여공간 투영을 $P_C$, $F_\pm=\mathrm{diag}\,(p_\pm)-p_\pm p_\pm^T$라 하면

$$
a_h=P_C(h_+-h_-),\qquad
b_h=P_CW^T\left[\delta p+\tfrac12(F_++F_-)\delta z\right],
$$

$$
u=\frac{a_h}{\|a_h\|},\qquad
b_\perp=b_h-u(u^Tb_h),\qquad v=\frac{b_\perp}{\|b_\perp\|},
$$

$$
Q(\theta)=I+(\cos\theta-1)(uu^T+vv^T)
+\sin\theta(vu^T-uv^T).
$$

$a_h$와 $b_\perp$가 양의 노름을 가지는 경우 $\mathcal D'(0)=\|a_h\|\|b_\perp\|>0$다. 이 미분은 고정한 두 hidden에 대한 구별도의 해석적 각도 미분이다. 유한 회전은 실제 readout에서 측정한다. $QC=C$, $Q^TQ=I$, $W'W'^T=WW^T$와 두 출력 점수의 중간값을 실수 연산에서 보존한다. 각도와 가중치 수정 예산은 기존 readout 계약을 적용한다. 중간값을 고정하므로 중심 상태를 이동시켜 구별도를 높이는 자유도가 제거된다.

제품은 이 방향을 기존 반각 좌표의 부호 규약으로 표현한다. 일차 방향이 퇴화하면 확률이 높은 출력 행을 기준으로 최대 32개 행 차이를 보호 공간에 투영해 유한 후보를 만든다. 실제 구별도 증가와 보호 조건으로 채택한다. 각도 상한은 가중치 Frobenius 예산으로 추가 제한하고, 상한의 0.04·0.2·1배에서 실제 F32 회전·readout을 측정한다. 입력 기준 상태, 선택 방향에 직교한 대조 주입의 두 상태, 입력 파일의 추가 보호 벡터를 보호한다. 출력 중간값의 실제 변화량도 측정한다. version 3 수정안은 이 구별도 목적, 수치 입력 SHA256, 가중치·실행 설정 해시를 담는다. `probe --inputs ... --proposal ...`은 저장한 입력에서 전체 decoder를 재실행한다.

발상에 사용한 자료는 [당구 반사광의 초점선](https://arxiv.org/abs/2112.07852), [열교환기의 로그평균 온도차](https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node131.html), [유한 Markov 연쇄의 엔트로피 흐름](https://www.janmaas.org/papers/discrete.pdf)이다. 위 주입 분포, softmax의 유한 차이 행렬, 잔차 분해와 보호 회전의 구별도 미분은 이 프로젝트의 연산 계약에서 유도한다.

## 보호 공간과 회전

보호할 내부 벡터를 열로 모은 행렬을 $C$라 두고, 그 열공간의 직교 여공간에서 단위 직교 벡터 $u,v$를 선택한다.

$$
u^TC=v^TC=0,\qquad P=uu^T+vv^T,\qquad J=uv^T-vu^T.
$$

반각 좌표 $t=\tan(\theta/2)$를 사용한다.

$$
Q(t)=I-\frac{2t^2}{1+t^2}P+\frac{2t}{1+t^2}J.
$$

$J^T=-J$, $J^2=-P$, $P^2=P$로부터 $Q^TQ=I$를 얻는다. 따라서 실수 연산에서 다음 항등식이 성립한다.

$$
W'C=WC,\qquad W'W'^T=WW^T,\qquad
\mathrm{rank}\,(W'-W)\le2.
$$

보호 접두사의 전체 vocabulary 점수와 출력 행벡터 사이의 내적이 유지된다. 회전은 $u,v$가 만드는 평면에서만 작용한다. [`src/rotation.rs`](src/rotation.rs)는 두 번의 직교 투영으로 보호 공간의 기저를 구성하며 상대 rank 기준은 $10^{-10}$이다.

## 후보 평면

### 평면 선택

목표 token $y$와 현재 가장 높은 경쟁 token $j$의 출력 행 차이를 $d=w_y-w_j$, 관측 margin을 $m$, 요구 margin을 $\tau$라 둔다. 부족량 $\lambda=\max(\tau-m,0)$을 가중치로 사용한다. 보호 공간의 여공간으로 투영한 $\bar d,\bar h$로 반대칭 연산자를 만든다.

$$
K=\sum_i\lambda_i(\bar d_i\bar h_i^T-\bar h_i\bar d_i^T).
$$

관측 벡터들이 생성하는 공간에서 $K^TK$의 고유벡터 $u$를 구하고 $v=-Ku/\|Ku\|$로 회전 평면을 구성한다. 가장 큰 고유값의 평면은 정규화된 평면 회전 중 가중 margin 합의 일차 증가량을 최대화한다. 이어지는 고유벡터 평면과 개별 $(\bar d_i,\bar h_i)$ 평면도 후보로 만들며 같은 평면은 한 번만 검사한다. `max_planes`가 후보 수를 정한다.

이 단계는 수정할 평면을 정한다. 각 평면의 각도는 다음 유한 회전 조건으로 계산한다.

### 관측 공간의 선형 응답

규모 편차의 개선 요구량을 여러 관측에 동시에 배분하는 확장식은 관측 공간에서 계산한다. 증가시킬 관측값을 $f_i$, 보호할 관측값을 $p_j$라 하고, 고정한 수정 좌표에서 그 미분을 행으로 쌓은 행렬을 $J,B$라 둔다. 손실에는 $f_i=-L_i$, token 조건에는 margin을 사용한다. 보호 투영과 응답 행렬은

$$
\Pi=I-B^T(BB^T)^\dagger B,\qquad
K=J\Pi J^T.
$$

$r\in\mathrm{range}\,(K)$인 선형 조건 $J\Delta w=r,\ B\Delta w=0$에서

$$
\lambda=K^\dagger r,\qquad
\Delta w=\Pi J^T\lambda,\qquad
\|\Delta w\|^2=r^TK^\dagger r
$$

가 해당 좌표의 최소 유클리드 수정이다. 보호 항을 포함한 전체 관측 Gram을 만든 뒤 $K=G_{ff}-G_{fp}G_{pp}^\dagger G_{pf}$로 계산하면 전역 투영 $\Pi$를 저장하는 비용을 줄인다. $\lambda$는 함께 만족시킬 관측들의 응답을 조정한다.

affine 가중치 $W_\ell$의 입력을 $x_{\ell i t}$, 관측값에서 역으로 전달한 미분을 $d_{\ell i t}$라 두면

$$
\nabla_{W_\ell}f_i=\sum_t d_{\ell i t}x_{\ell i t}^T,
$$

$$
G_{ij}^{(\ell)}
=\sum_{t,u}
(d_{\ell i t}^Td_{\ell j u})
(x_{\ell i t}^Tx_{\ell j u}).
$$

이 항을 층마다 합하면 명시적인 가중치 Jacobian과 같은 Gram을 얻는다. 같은 tensor를 여러 위치에서 사용하면 모든 사용 위치를 합산한다. 정규화·embedding 등 다른 가중치 연산은 해당 미분 계약의 내적 항을 더한다. 이 식은 전 층 미분을 연결할 때의 계산 계약이다. 제품의 응답 구현은 아래의 readout 반대칭 좌표를 사용한다.

### Readout의 공동 회전

현재 readout 회전에는 같은 응답식을 직접 적용할 수 있다. 보호 공간에 직교 투영한 $\bar d_i,\bar h_i$로 $A_i=\bar d_i\bar h_i^T-\bar h_i\bar d_i^T$를 만들고, 반대칭 행렬의 내적을 $\langle A,B\rangle=\tfrac12\mathrm{tr}\,(A^TB)$로 정하면

$$
K_{ij}
=(\bar d_i^T\bar d_j)(\bar h_i^T\bar h_j)
-(\bar d_i^T\bar h_j)(\bar h_i^T\bar d_j),\qquad
\Omega=\sum_i(K^\dagger r)_iA_i.
$$

$\Omega$는 요구한 margin 변화를 만드는 최소 각도 노름의 일차 수정이다. $\Omega$의 상은 관측 벡터들의 생성 공간에 들어가며 rank는 최대 $2m$이다. 그 공간의 직교 기저 $U$와 $\omega=U^T\Omega U$를 사용하면 유한 수정은

$$
Q_U(t)=(I-t\omega/2)^{-1}(I+t\omega/2),\qquad
Q(t)=I+U(Q_U(t)-I)U^T
$$

로 계산한다. 실수 연산에서 직교성과 보호 공간을 유지한다. [`src/response.rs`](src/response.rs)는 관측 Gram의 의사역행렬로 공동 수정 방향을 구한다. [`src/diagnosis.rs`](src/diagnosis.rs)는 평면 후보의 검사를 마친 뒤 이 방향을 적용하고 압축 GPU 출력의 margin이 개선되는 보폭을 선택한다. 각 단계에서 누적 회전의 최대 주각과 가중치 수정 노름 상한을 적용한다. 원본 성공 경로와 이미 요구 margin을 만족한 목표 접두사의 벡터를 기저에서 제외해 고정한다. 새로운 경쟁 token은 다음 단계의 응답식에 반영한다. 선형 응답의 잔차와 최종 원본 readout 측정값은 후보 기록에 각각 저장한다.

### 출력과 비용의 압축

공동 탐색의 연산 축소는 고정한 보호 기저 $U\in\mathbb R^{d\times r}$, 국소 회전 $R\in SO(r)$, 관측 좌표 $x=U^Th$를 사용한다. $F=WU$, $G=F^TF$, 원본 점수 $z_0=Wh+b$를 한 번 구하면

$$
z_R=z_0+F(R-I)x,\qquad
\mathcal E(R)=\|W'-W\|_F^2
=\mathrm{tr}\,((R-I)^TG(R-I))
=2\mathrm{tr}\,G-2\mathrm{tr}\,(GR).
$$

$U^TU=I$와 $R^TR=I$가 이 항등식의 조건이다. 기저와 원본 가중치를 고정한 동안 $F,G,z_0,x$를 공유한다. 첫 Gram 구성은 $O(Vr^2)$, 이후 안정적인 이차형식 비용은 $O(r^3)$, trace 비용은 $O(r^2)$다. 출력 평가의 행렬 폭은 $d$에서 $r$로 줄어든다. 작은 회전의 비용에는 이차형식을 사용해 양의 작은 변화량을 직접 계산한다. GPU F32의 곱 순서와 저장 반올림 오차는 원본 readout 측정값과 대조한다.

### Cayley 연립방정식의 축소

공동 방향 $\Omega=\sum_i\lambda_i(d_ix_i^T-x_id_i^T)$는 $a_i=\sqrt{|\lambda_i|}d_i$, $b_i=\mathrm{sgn}\,(\lambda_i)\sqrt{|\lambda_i|}x_i$, $S=[a_1,\ldots,a_m,b_1,\ldots,b_m]$, $J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}$로 $\Omega=SJS^T$가 된다. $K=S^TS$라 두면

$$
T_t=t(I-tJK/2)^{-1}J,\qquad
\mathrm{Cayley}\,(t\Omega)=I+ST_tS^T,\qquad
R_t=R+RST_tS^T.
$$

이는 $(I-aSJS^T)S=S(I-aJS^TS)$에서 얻는 정확한 축소다. 행렬식도 $\det(I-aSJS^T)=\det(I-aJS^TS)$이며, 반대칭 $\Omega$의 켤레 고유값 쌍 때문에 실수 보폭에서 이 값은 양수다. $S$의 열이 선형 종속인 경우에도 식이 성립한다. $2m<r$인 후보군은 작은 $2m$차원 연립방정식을 공유할 수 있다. $D=R-I$, $P=S^TD^TGRS$, $N=S^TR^TGRS$를 보폭 탐색 전에 준비하면

$$
\mathcal E(R_t)=\mathcal E(R)+2\mathrm{tr}\,(PT_t)
+\mathrm{tr}\,(T_t^TNT_tK).
$$

초기 $R=I$에서는 마지막 양의 이차항이 수정 비용을 직접 준다. 후보별 비용 계산에 필요한 행렬 크기는 $2m$이다. `ResponsePath`는 $2m<r$에서 $S,J,K,P,N$과 기준 비용을 단계마다 한 번 구성하고 다섯 보폭이 공유한다. $2m\ge r$이면 $r$차원 Cayley 식을 푼다. 증분 비용이 항별 크기로 계산한 반올림 범위에 들어가면 `WeightMetric`의 안정적인 이차형식으로 평가한다. 누적 회전의 최대 주각은 현재 각도 계약으로 판정한다.

### 같은 관측을 유지하는 최소 비용 완성

관측 공간에서의 다른 축소는 같은 관측 결과를 만드는 회전들의 등거리 확장이다. 모든 변화 대상 좌표를 모은 $H$의 열공간 기저를 $E\in\mathbb R^{r\times k}$, 그 이동을 $Y=RE$라 두면 출력 조건은 $Y$로 결정된다. $Y^TY=I$이며, 관측 이동의 자유도는 $rk-k(k+1)/2$다. $0<k<r$에서 $[E,E_\perp]$, $[Y,Y_\perp]$를 방향을 보존하는 직교 기저로 완성하면 모든 같은 관측 회전은

$$
R_B=YE^T+Y_\perp B E_\perp^T,\qquad B\in SO(r-k)
$$

로 표현된다. $M=E_\perp^TGY_\perp=L\Sigma V^T$, $\sigma_1\ge\cdots\ge\sigma_{r-k}$, $\delta=\det(VL^T)$라 두면

$$
B_*=V\mathrm{diag}\,(1,\ldots,1,\delta)L^T,
$$
$$
\min_B\mathcal E(R_B)
=2\mathrm{tr}\,G-2\mathrm{tr}\,(E^TGY)
-2\left(\sum_{i<r-k}\sigma_i+\delta\sigma_{r-k}\right).
$$

증명은 $\mathrm{tr}\,(GR_B)=\mathrm{tr}\,(E^TGY)+\mathrm{tr}\,(MB)$를 분리하고 방향 제약을 둔 특이값 trace 최댓값을 구하는 것이다. 완성한 $R_{B_*}$가 기존 각도 상한 안에 있으면 같은 관측 점수와 최소 가중치 수정 비용을 함께 얻는다. 보호 상태는 $U$에 직교하는 조건으로 유지한다. `ObservedSpace`는 변화 대상 좌표의 특이값이 $\max(10^{-12},10^{-9}\sigma_{\max})$보다 큰 방향을 선택해 관측 열공간을 탐색 시작 시 고정한다. 관측 rank가 0이면 항등 회전, $r$이면 주어진 회전이 최솟값을 준다.

[`src/response.rs`](src/response.rs)는 캐시한 비용, 축소 Cayley, 최소 비용 완성을 계산한다. [`src/diagnosis.rs`](src/diagnosis.rs)의 `CompactReadout`은 $F$와 기준 점수를 GPU F32로 한 번 적재하고, F64로 계산한 $(R-I)x$를 F32로 변환해 공통 `Gpu::linear`로 평가한다. 탐색의 최종 회전과 완성 후보는 모든 관측 접두사에서 원본 GPU readout으로 측정한다. 완성 후보가 각도·비용 상한, 모든 접두사의 argmax, 이미 만족한 목표 margin, 보호 출력 오차를 유지하고 두 후보 사이의 scaled logit 차이가 허용 오차 이내이면 완성을 채택한다. 이 판정은 수치 rank 절단과 F32 연산 순서의 영향을 포함한다. 기존 후보가 전체 채택 조건을 만족하면 완성 후보에도 같은 조건을 적용한다. 결과의 `completion`은 비용 전후·관측 rank·실제 출력 차이·적용 여부를 기록한다. `computation`은 기저 출력·압축 출력·최종 검증 호출 수, 비용 구성 횟수, 회전 해법별 시행 수와 공동 준비·탐색·완성 대수 계산 시간을 기록한다.

발상의 수학적 출처는 [이차형식의 Witt 등거리 확장](https://swc-math.github.io/aws/2009/09HankeNotes.pdf)과 [Ihara 제타함수의 연산 공간 축소](https://www.math.lsu.edu/system/files/jwh2002b.pdf)다. 위 비용식·응답 축소·최소 수정 완성식은 프로젝트의 readout 계약에서 유도했다.

발상에 사용한 자료는 [전기 회로의 유효 저항과 응답 행렬](https://www.cs.yale.edu/homes/spielman/462/2007/lect9-07.pdf), [충격 신호 복원의 소거 필터](https://bigwww.epfl.ch/publications/vetterli0201.html)다. 규모 편차, 보호 응답, 회전 Gram은 위 계산 계약에서 직접 유도한다.

## 모든 경쟁 token에 대한 이차부등식

고정 접두사에서 정답과 경쟁 token의 원본 점수 차이를 $m$이라 두면 bias는 이미 $m$에 포함된다. 각 경쟁 token에 대해

$$
\alpha=d^TJh,\qquad \beta=d^TPh,
$$

$$
m(t)=m+\frac{2\alpha t-2\beta t^2}{1+t^2}.
$$

$1+t^2>0$이므로 $m(t)\ge\tau$는 정확히 다음 조건과 동치다.

$$
(m-\tau-2\beta)t^2+2\alpha t+(m-\tau)\ge0.
$$

[`src/diagnosis.rs`](src/diagnosis.rs)는 평면마다 두 출력 방향을 한 번씩 계산하고 접두사의 두 투영값으로 모든 계수를 구성한다. [`src/interval.rs`](src/interval.rs)는 각 조건이 금지하는 열린 구간을 합쳐 모든 목표 token과 전체 vocabulary 경쟁 token의 공통 해를 유지한다.

가중치 변화량은

$$
\|WQ-W\|_F^2=
\frac{4t^2}{1+t^2}\bigl(\|Wu\|^2+\|Wv\|^2\bigr).
$$

고정 평면에서 이 값은 $|t|$에 따라 증가한다. 따라서 교집합에서 0에 가장 가까운 $t$가 그 평면의 최소 변화량 해다. 여러 평면의 후보는 이 변화량 순으로 검증한다. 최소성의 범위는 선택한 각 평면의 허용 회전각이다.

### 금지 구간과 해집합

같은 해를 계산하는 금지 구간 표현은 다음과 같다. $c_i=m_i-\tau$, $a_i=c_i-2\beta_i$, $b_i=2\alpha_i$라 두면, $q_i(t)=a_it^2+b_it+c_i$가 음수인 열린 구간들이 해당 조건의 금지 영역이다. 각도 좌표에서는

$$
m_i(\theta)-\tau=c_i-\beta_i+\beta_i\cos\theta+\alpha_i\sin\theta
$$

이므로 하나의 조건은 원둘레의 금지 호를 정한다. 반각 좌표에서는 최대 두 개의 열린 구간으로 표현한다. 금지 구간의 합집합을 $U$, 반각 범위를 $[-T,T]$라 두면 해집합은 $F=[-T,T]\setminus U$다. 0이 $U$ 안에 있으면 0을 포함하는 연결 성분의 두 끝점 중 범위 안에서 절댓값이 가장 작은 끝점이 최소 수정 해다. 끝점에서 닿는 두 열린 구간은 그 점을 가능한 해로 남긴다.

남은 해집합의 양 끝점을 $L=\min F$, $R=\max F$로 관리하면 조건을 더 줄일 수 있다. $[L,R]$ 전체에서 $q_i\ge0$이면 이후의 해집합에서도 그 조건이 성립한다. 이 확인에는 두 끝점과 구간 안의 꼭짓점만 필요하다. $a_i>0$이고 $-b_i/(2a_i)\in[L,R]$이면 꼭짓점 조건은 $4a_ic_i-b_i^2\ge0$이다. 이외에는 두 끝점의 부호로 판정한다. 구간을 실제로 자르는 조건만 근을 계산해 합집합에 반영한다.

균형 검색 트리에 서로 겹치는 금지 구간을 합쳐 저장하면 전체 N개 조건과 생성된 K개 금지 구간의 계산량은 O(N + K log(K+1)), 저장량은 O(K)다. 병합으로 제거되는 구간은 한 번만 제거된다. [`src/interval.rs`](src/interval.rs)는 이 표현과 양 끝점·꼭짓점 판정을 사용한다. 계수는 2의 거듭제곱으로 정규화하며, 서로 닿는 열린 구간 사이의 단일점과 대칭 최소해를 보존한다. 제곱근 계산 횟수와 사전 판정 횟수를 보고서에 기록한다.

발상의 참고 자료는 [해시계의 원 투영과 그림자](https://www.dezonnewijzerkring.nl/downloads/JdR-EquatorProj.pdf), [원둘레를 임의의 호로 덮는 문제](https://statistics.stanford.edu/technical-reports/covering-circle-random-arcs-random-sizes)다. 위의 금지 구간과 비용식은 현재 프로그램의 margin 식을 직접 변환해 얻는다.

## 실제 실행의 검증

### 정밀도와 채택 조건

평면·기저·부등식은 F64, decoder와 회전 커널은 F32로 계산한다. 원본 F32/F16/BF16 가중치를 그대로 읽는다. 부등식의 목표에는 $8\epsilon\max(1,\max|\mathrm{logit}|)$의 수치 여유를 더한다. 여기서 $\epsilon$은 사용자 검증 허용 오차다.

후보마다 GPU에서 다음을 측정한다.

1. 목표 접두사마다 정답 token과 전체 경쟁 token 사이의 실제 최소 margin.
2. 보호 접두사의 전체 출력 변화량과 $1+|원본\ 점수|$로 나눈 변화량.
3. 관측 접두사의 argmax가 이어지는 전체 답변과 EOS 종료.
4. 보호 사례의 원본 생성 token 열 유지와 실패 사례의 목표 token 열 일치.

실제 margin이 요구값 이상이고 보호 오차가 허용값 이내이며 관측 경로의 token이 EOS까지 일치하면 수정안을 채택한다. 고정 decoder에서 매 목표 token이 양의 margin으로 선택되고 마지막 EOS도 선택되면 접두사에 대한 귀납으로 전체 목표 답변의 greedy 생성이 따른다. 전체 모델 회귀 검사는 이 판정과 수정안을 재적용한 생성을 대조한다.

### 접두사 검증과 전체 생성

고정 접두사 $p_i$의 내부 벡터를 $h_i$라 두면, readout 회전에 대해 $h_Q(p_i)=h_0(p_i)$가 성립한다. 따라서 관측한 $h_i$에 실제 GPU 회전과 원본 readout을 실행한 $z_i(Q)=WQh_i+b$가 그 접두사의 실제 수정 출력이다. 첫 목표 token이 선택되면 다음 접두사가 관측 경로와 같아지고, 이 관계를 EOS까지 반복하면 전체 생성이 확인된다. 보호 사례에도 원본 생성 경로의 모든 접두사에서 점수 변화와 선택 token을 검사해 같은 귀납을 적용한다.

[`src/diagnosis.rs`](src/diagnosis.rs)는 각 관측의 prompt·token 순서·내부 벡터를 함께 유지한다. 후보별 검증은 해당 벡터에 실제 GPU 회전·readout·argmax를 실행한다. 모든 token이 일치하면 전체 경로를 확인하고, 첫 불일치가 생기면 그 지점까지의 생성 접두사를 기록한다. 후보 검증 함수는 모델을 읽기 참조로 받아 decoder 상태를 고정한다. 평면 수정은 기존 F32 회전 연산 순서를 사용한다. 공동 수정은 기저 투영, 작은 행렬 곱, 기저 복원으로 실행한다. `generate` 명령은 저장한 수정안을 전체 decoder 실행에 재적용한다.

### 개입 경로의 기여 분석

수정의 가중치별 귀속은 명시한 개입 경로 $\theta(t)$에서 측정한다. 규모 편차 $R_k$의 좌표별 기여량을

$$
C_a=-\int_0^1
\frac{\partial R_k(\theta(t))}{\partial\theta_a}
\dot\theta_a(t)\,dt
$$

로 정의하면 연쇄법칙에 의해 $\sum_a C_a=R_k(\theta(0))-R_k(\theta(1))$이다. 모든 수정 좌표를 포함하며, 편차를 증가시킨 기여는 음수로 보존한다. 경로와 가중치 묶음을 실행 조건에 포함한다. 직선 경로에서는 $\dot\theta=\Delta\theta$이고, 고정 readout 접두사의 손실 미분은 $(\mathrm{softmax}\,(z)-e_y)h^T$이다. $\Delta WC=0$인 두 끝점의 직선 경로는 보호 점수를 유지한다. 직교성과 출력 행 Gram의 보존은 회전 경로와 최종 회전에 적용한다.

[`src/attribution.rs`](src/attribution.rs)는 readout 직선 경로의 softmax 미분을 적분하고 기여 합과 실제 손실 감소량의 오차를 기록한다. 적분은 합의 잔차가 허용값에 도달하면 끝내며, 최대 분할 수에서도 측정한 잔차와 수렴 상태를 남긴다. 출력 행의 기여는 실제 F32 끝점 점수를 사용한다. 해당 행 안의 좌표 기여는 F64 가중치 변화량으로 복원하고 행 기여와의 차이를 함께 기록한다. 기여가 큰 행의 수정을 원래 값으로 되돌린 손실도 계산한다. 이 행 되돌림은 기여 분석의 유한 개입 조건이다. 채택하는 수정안은 공동 회전의 전체 가중치 상태다.

규모 보고서는 각 checkpoint의 기여에 해당 규모 대비 계수를 곱한다. 행·열 좌표는 checkpoint별 tensor 이름과 함께 기록하며, 표시한 행과 나머지 행의 기여를 합쳐 전체 편차 회복량과 대조한다. 행 되돌림에 따른 편차 증가량도 같은 계수로 계산한다. 전역 응답식의 보호 조건은 일차 미분 조건이고, 제품의 readout 회전은 보호 공간을 고정하는 유한 변환이다.

저장한 수정안에는 회전 벡터·반각 좌표·readout 연결·모델 revision·모든 입력 파일의 해시를 담는다. 원본 checkpoint는 읽기 전용으로 적재하고 수정 상태는 실행 중 별도로 소유한다.

## 구성과 수명

[`src/hub.rs`](src/hub.rs)는 revision과 원본 파일을 확보하고 해시를 계산한다. [`src/graph.rs`](src/graph.rs)는 설정과 tensor 형상으로 decoder 연산 계약을 확정한다. [`src/text.rs`](src/text.rs)는 원본 tokenizer·chat template·고정 prompt와 정답 continuation을 준비한다. [`src/probes.rs`](src/probes.rs)는 사용자 입력과 계산 가능한 기호 사례를 제공한다.

[`src/model.rs`](src/model.rs)는 토큰·수치 입력의 공통 decoder, KV 캐시, 잔차 분기·최종 내부 벡터 관측, 별도 projection 개입과 readout 상태를 실행한다. [`src/gpu.rs`](src/gpu.rs)는 tensor·커널 인수·컴파일 metadata를 관리하고 [`src/device.rs`](src/device.rs)는 CUDA·HIP의 메모리·context·module 수명을 관리한다. [`kernels/ops.py`](kernels/ops.py)와 [`kernels/compile.py`](kernels/compile.py)는 Triton 연산과 컴파일 연결을 담당한다. 커널 소스와 실행 대상의 내용 해시로 캐시를 구성한다.

[`src/rotation.rs`](src/rotation.rs)는 보호 공간과 평면 후보를, [`src/interval.rs`](src/interval.rs)는 금지 구간 해법을, [`src/response.rs`](src/response.rs)는 공동 응답과 작은 회전을 계산한다. [`src/diagnosis.rs`](src/diagnosis.rs)는 접두사 관측·후보 계산·GPU readout 검증·수정안 저장을 담당한다. [`src/contrast.rs`](src/contrast.rs)는 유한 확률·정규화 차이와 구별도 미분을, [`src/injection.rs`](src/injection.rs)는 수치 입력 생성·방향 탐색·상쇄 좌표·유한 개입·보호 회전과 재실행을 담당한다. [`src/attribution.rs`](src/attribution.rs)는 가중치 기여와 행 되돌림 효과를, [`src/scaling.rs`](src/scaling.rs)는 checkpoint 계열의 편차와 회복량을 계산한다. [`src/main.rs`](src/main.rs)는 모델 확보·입력 생성·진단·규모 분석·수정안 재적용 명령을 제공한다. 실행 결과와 컴파일 산출물은 프로젝트 외부에 저장한다.
