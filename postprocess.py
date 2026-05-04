# postprocess.py
import numpy as np
import hyperparameters as hp 
import torch 
import params as p

def postprocess(input_img, range_preds, u, v, ranges, device, k=5):
    """
    Runs KNN voting in 3D to clean up label bleeding.

    Args:
        input_img:    (5, H, W) representing range, x, y, z, and remission for each
                        projected coordinate (u,v)
        range_preds:  (H, W)  predicted class labels on range image
        u, v:         (N,)    pixel coords each point projected to
        k:            number of neighbors for voting
        ranges:       (N,) ranges R for each point 

    Returns:
        final_preds:  (N,)  refined per-point class labels
    """
    H, W = range_preds.shape
    N = len(u)
    I_range = torch.as_tensor(input_img[0, 0, :, :], device=device) #range image of size W x H 
    I_label = torch.from_numpy(range_preds).to(device).long() #label image of predictions of size W x H
    R = torch.as_tensor(ranges, device=device).float()

    #0: pad image so the neighbor extraction doesn't exceed the boundaries 
    S = hp.NBRHOOD_SIZE
    pad = S // 2
    padded = torch.nn.functional.pad(I_range, (pad, pad, pad, pad), mode="constant", value=0) #experiment with different types of padding
    padded_labels = torch.nn.functional.pad(I_label, (pad, pad, pad, pad), mode="constant", value=0)

    #1. create a [h*w, S^2] matrix containing unwrapped version of SxS neighborhood around each point 
    # Each column contains unwrapped version of neighborhood, and the column center contains the actual pixel's range 
    nbrs = torch.zeros(H * W, S * S, device=device, dtype=I_range.dtype) #output; N' in the algo

    # print("padded shape:", padded.shape)
    # print("H, W, S:", H, W, S)
    # #breakpoint()

    for vp in range(H): 
        for up in range(W): 
            pixel_idx = vp * W + up
            patch = padded[vp: vp + S, up: up + S]

            #flatten to S^2 
            nbrs[pixel_idx, :] = patch.reshape(-1)

    #2. Extend representation to a matrix of dim [S^2, N] w/ range neighborhood of all can points 
    # Do this by indexing the unfolded image matrix 
    N_matrix = torch.zeros(N, S*S, device=device, dtype=I_range.dtype)

    for i in range(N): 
        ui = u[i]
        vi = v[i]

        pixel_idx = vi * W + ui #get the row index in nbrs

        #copy the neighborhood from nbrs
        N_matrix[i, :] = nbrs[pixel_idx, :]

    #3. Replace center row of matrix with actual range readings for each point 
    # Result: [S^2, N] matrix w/ range readings for points in the center row, and each column has SxS neighborhood
    center_idx = (S * S - 1) // 2
    N_matrix[:, center_idx] = R

    #4. Reshape label matrix to
    Lp = torch.zeros(H * W, S * S, device=device, dtype=torch.long) #output; L' in the algo

    for vp in range(H): 
        for up in range(W): 
            pixel_idx = vp * W + up
            patch = padded_labels[vp: vp + S, up: up + S]

            #flatten to S^2 
            Lp[pixel_idx, :] = patch.reshape(-1)

    L_matrix = torch.zeros(N, S*S, device=device, dtype=torch.long)

    for i in range(N): 
        ui = u[i]
        vi = v[i]

        pixel_idx = vi * W + ui #get the row index in nbrs

        #copy the neighborhood from nbrs
        L_matrix[i, :] = Lp[pixel_idx, :]

    #5. Compute distance to neighbors D for each point
    D = torch.abs(N_matrix - R[:, None])

    #6. Weigh the distances by inverse Gaussian kernel  
    coords = torch.arange(S, device=device).float() 
    coords = coords - (S - 1) / 2.0 # center it around 0

    #make the gaussian
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-(xx ** 2 + yy ** 2) / (2 * hp.SIGMA ** 2))
    gaussian = gaussian.reshape(-1)

    G_max = gaussian.max()
    G = 1.0 - gaussian / G_max #inverse gaussian

    D_weighted = D * G[None, :]

    #7. Find k closest points among S^2 candidates to get indexes for top k
    knn_distances, knn_indices = torch.topk(
        D_weighted,
        k=k,
        dim=1,
        largest=False
    )

    #8. Gather votes from all the labels of points within the cutoff threshold
    #done via gather add operation
    #result:    [C, N] matrix; C = num of classes; each row contains number of votes in its index class
    L_knn = torch.gather(L_matrix, dim=1, index=knn_indices) 

    #make sure to get rid of points beyond the cutoff
    L_knn = torch.where(
        knn_distances > hp.CUT_OFF,
        torch.full_like(L_knn, -1),
        L_knn
    )

    #9. Accumulate votes V (shape (N, num_classes))
    V = torch.zeros((N, p.NUM_CLASSES), device=device)

    valid_mask = L_knn >= 0
    point_indices = torch.arange(N, device=device)[:, None].expand(-1, k)
    #breakpoint()
    V[point_indices[valid_mask], L_knn[valid_mask].long()] += 1

    #10. argmax over columns to get [1,N] vector with labels for each point in the input
    #this is the output! 
    L_consensus = torch.argmax(V, dim=1)

    return L_consensus #(N,)
