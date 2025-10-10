# ============================================================================
# COPY THIS ENTIRE forward() METHOD TO REPLACE THE ONE IN YOUR CELL 5
# ============================================================================

def forward(self, coords_decoding, tokens, coords_modulation=None):
    """
    Forward pass with SCENT-style architecture - FIXED SPATIAL BIAS
    """
    B, query_shape = coords_decoding.shape[0], coords_decoding.shape[1:-1]
    coords_dec = coords_decoding.view(B, -1, coords_decoding.shape[-1])

    if coords_modulation is not None:
        coords_mod = coords_modulation.view(B, -1, coords_modulation.shape[-1])
    else:
        coords_mod = coords_dec

    # ========== FIX STARTS HERE ==========
    # Use actual coordinate grid dimensions (not self.patch_num!)
    grid_mod = coords_mod[0]  # (HW, 2)
    num_queries = grid_mod.shape[0]  # e.g., 1024 for 32×32
    H_mod = W_mod = int(math.sqrt(num_queries))  # e.g., 32 for 32×32

    # Now compute bias using actual grid size
    indexes = self.get_patch_index(grid_mod, H_mod, W_mod)
    rel_distances = self.approximate_relative_distances(
        indexes, H_mod, W_mod, tokens.shape[1]
    )
    bias = repeat(rel_distances, 'l n -> b l n', b=B)
    # ========== FIX ENDS HERE ==========

    # Query encoding
    x_q = repeat(
        gaussian_fourier_encode(coords_mod[0], self.B_q), 'l d -> b l d', b=B
    )
    x_q = self.act(self.query_lin(x_q))

    # Extract modulation
    modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)

    # === MULTI-SCALE DECODING (SCENT-STYLE) ===

    modulations_l = []
    fourier_encodings = []

    for k in range(self.layer_num):
        # Bandwidth encoding
        x_l_fourier = gaussian_fourier_encode(coords_dec[0], self.B_ls[k])
        x_l_fourier_batch = repeat(x_l_fourier, 'l d -> b l d', b=B)
        fourier_encodings.append(x_l_fourier_batch)

        # Process through Attention + FF blocks
        h_l = self.apply_block_sequence(x_l_fourier_batch, self.bandwidth_lins[k])

        # Modulation projection
        m_proj = self.apply_block_sequence(modulation_vector, self.modulation_lins[k])

        # Combine
        m_l = self.act(h_l + m_proj)
        modulations_l.append(m_l)

    # Residual connections between scales
    h_v = [modulations_l[0]]
    for i in range(self.layer_num - 1):
        x_combined = modulations_l[i+1] + h_v[i]

        # Apply Attention + FF
        attn, ff = self.hv_lins[i][0], self.hv_lins[i][1]
        x_combined = attn(x_combined) + x_combined
        x_combined = ff(x_combined) + x_combined

        h_v.append(x_combined)

    # Output with SKIP CONNECTIONS
    outs = []
    for i in range(self.layer_num):
        # Add skip connection from original Fourier encoding
        fourier_skip = self.fourier_skip_projs[i](fourier_encodings[i])
        out_with_skip = self.out_lins[i](h_v[i] + fourier_skip)
        outs.append(out_with_skip)

    out = sum(outs)
    out = out.view(B, *query_shape, -1)

    return out

# ============================================================================
# INSTRUCTIONS:
# 1. In your Jupyter notebook Cell 5, find the forward() method
# 2. Delete the entire forward() method
# 3. Copy-paste THIS forward() method in its place
# 4. Re-run Cell 5
# 5. Then re-run the training cell
# ============================================================================
