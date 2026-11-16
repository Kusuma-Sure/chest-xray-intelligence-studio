# Deep Learning Model Documentation & Mathematical Formulations

This document provides the mathematical specifications for the four architectures and their explainability attribution mechanisms implemented in the Chest X-ray Intelligence Studio.

---

## 1. Vision Transformer (ViT-B/16)

### Architecture
- Input image $X \in \mathbb{R}^{H \times W \times C}$ reshaped into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $P = 16$ and $N = \frac{HW}{P^2} = 196$.
- Linear projection of flattened patches with prepended $[CLS]$ token and 1D learnable position embeddings:
  $$z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \dots; x_p^N E] + E_{\text{pos}}$$
- 12 Transformer Encoder blocks with Multi-Head Self-Attention (MSA) and MLP:
  $$z_\ell' = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}$$
  $$z_\ell = \text{MLP}(\text{LN}(z_\ell')) + z_\ell'$$
- Multi-label classification head maps $z_L^0$ ([CLS] representation) to 13 logits.

### ViT Attention Rollout Mathematics
To track how information flows from the input patches to the final $[CLS]$ token across all $L=12$ layers:
Let $A_\ell \in \mathbb{R}^{197 \times 197}$ be the raw multi-head self-attention matrix at layer $\ell$ averaged across all heads:
$$A_\ell = \frac{1}{H} \sum_{h=1}^H W_{\text{attn}, \ell, h}$$
To account for residual skip connections, identity matrix $I$ is added:
$$A_\ell' = 0.5 A_\ell + 0.5 I$$
The matrix is re-normalized such that rows sum to 1:
$$\hat{A}_\ell = D_\ell^{-1} A_\ell' \quad \text{where} \quad D_{\ell, ii} = \sum_j A_{\ell, ij}'$$
The recursive Attention Rollout $R_L$ is computed by forward matrix multiplication:
$$R_\ell = \hat{A}_\ell \cdot R_{\ell-1} \quad \text{with} \quad R_0 = I$$
The attribution map for the image patches is obtained from the $[CLS]$ token row (excluding token 0):
$$\text{AttnMap} = R_L[0, 1:197] \in \mathbb{R}^{196} \xrightarrow{\text{reshape}} 14 \times 14$$

---

## 2. Capsule Network (CapsNet) with Dynamic Routing

### Architecture
- Conv1-3 layers extract low-level spatial features from greyscale radiographs.
- **Primary Capsules**: Feature maps are partitioned into 16 capsule types of dimension 8 ($7 \times 7 \times 16 = 784$ primary capsules $u_i \in \mathbb{R}^8$).
- Primary capsules are squashed:
  $$v = \frac{\|s\|^2}{1 + \|s\|^2} \frac{s}{\|s\|}$$
- **Disease Capsules**: 13 output capsules $v_j \in \mathbb{R}^{16}$, each representing the presence and instantiation parameters of one of the 13 CheXpert findings.

### Dynamic Routing Agreement Mathematics
1. For each primary capsule $i$ and disease capsule $j$, prediction vector:
   $$\hat{u}_{j|i} = W_{ij} u_i$$
2. Routing logits $b_{ij}$ initialized to 0.
3. For $r = 1 \dots 3$ routing iterations:
   - Coupling coefficients:
     $$c_{ij} = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})}$$
   - Weighted total capsule input:
     $$s_j = \sum_i c_{ij} \hat{u}_{j|i}$$
   - Squashed output capsule:
     $$v_j = \text{squash}(s_j)$$
   - Logit agreement update:
     $$b_{ij} \leftarrow b_{ij} + \hat{u}_{j|i} \cdot v_j$$
4. Output probability for finding $j$ is the capsule Euclidean norm:
   $$p_j = \|v_j\| \in [0, 1)$$

---

## 3. Graph Attention Network (Image-Region GAT)

### Architecture
- ResNet18 convolutional backbone extracts spatial feature map $F \in \mathbb{R}^{B \times 512 \times 7 \times 7}$.
- The 49 spatial cells form an image-region graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where each node $v_i \in \mathcal{V}$ represents an anatomical sector.
- Linear projection: $h_i = W_p F_i \in \mathbb{R}^{128}$.
- Adjacency matrix $A \in \{0, 1\}^{49 \times 49}$ connects each node to its 8 spatial neighbors plus self-loop.

### Graph Attention Mathematics
For multi-head attention ($K$ heads):
1. Pairwise attention coefficients between connected nodes $i$ and $j \in \mathcal{N}_i$:
   $$\alpha_{ij}^k = \frac{\exp\left(\text{LeakyReLU}\left(a_{\text{src}}^{kT} W^k h_i + a_{\text{dst}}^{kT} W^k h_j\right)\right)}{\sum_{m \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(a_{\text{src}}^{kT} W^k h_i + a_{\text{dst}}^{kT} W^k h_m\right)\right)}$$
2. Feature aggregation with non-linearity:
   $$h_i' = \text{ELU}\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k h_j\right)$$
3. Regional attention importance for visualization:
   $$\text{Importance}_i = \frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} \alpha_{ij} \xrightarrow{\text{reshape}} 7 \times 7$$

---

## 4. Grad-CAM and Grad-CAM++

### DenseNet121 Grad-CAM
Targeting the final convolutional feature maps $A^k$ of `features.denseblock4`:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y_c}{\partial A_{ij}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

### CapsNet Grad-CAM++
Targeting `conv3` using 1st, 2nd, and 3rd order gradients for high-fidelity multi-instance localization:
Let $g_{ij}^k = \frac{\partial y_c}{\partial A_{ij}^k}$.
The alpha weights are calculated as:
$$\alpha_{ij}^{kc} = \frac{(g_{ij}^k)^2}{2(g_{ij}^k)^2 + \sum_{a,b} A_{ab}^k (g_{ab}^k)^3 + \epsilon}$$
$$w_k^c = \sum_i \sum_j \alpha_{ij}^{kc} \cdot \text{ReLU}(g_{ij}^k)$$
$$L_{\text{Grad-CAM++}}^c = \text{ReLU}\left(\sum_k w_k^c A^k\right)$$
