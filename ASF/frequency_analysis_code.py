# Add this cell to your notebook to verify the resampling behavior

import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

def analyze_frequency_spectrum(model, test_images, device):
    """
    Analyze frequency content to verify if super-resolution
    generates new high frequencies or just resamples
    """
    model.eval()

    # Get one test image
    img_32 = test_images[0:1].to(device)

    with torch.no_grad():
        # Original 32×32
        coord_32 = create_coordinate_grid(32, 32, device).unsqueeze(0)
        recon_32 = model(img_32, coord_32)[0].cpu()

        # Super-resolved 64×64
        coord_64 = create_coordinate_grid(64, 64, device).unsqueeze(0)
        recon_64 = model(img_32, coord_64)[0].cpu()

        # Super-resolved 128×128
        coord_128 = create_coordinate_grid(128, 128, device).unsqueeze(0)
        recon_128 = model(img_32, coord_128)[0].cpu()

        # Bicubic upsampling for comparison
        bicubic_64 = F.interpolate(img_32, size=64, mode='bicubic', align_corners=False)[0].cpu()
        bicubic_128 = F.interpolate(img_32, size=128, mode='bicubic', align_corners=False)[0].cpu()

    # Compute 2D FFT for each resolution and channel (use red channel)
    def get_spectrum(img_tensor):
        """img_tensor: (H, W, 3) or (3, H, W)"""
        if img_tensor.shape[-1] == 3:  # (H, W, 3)
            img = img_tensor[..., 0].numpy()
        else:  # (3, H, W)
            img = img_tensor[0].numpy()

        spectrum = np.abs(fftshift(fft2(img)))
        return spectrum

    # Get spectra
    spec_32 = get_spectrum(recon_32)
    spec_64_model = get_spectrum(recon_64)
    spec_128_model = get_spectrum(recon_128)
    spec_64_bicubic = get_spectrum(bicubic_64)
    spec_128_bicubic = get_spectrum(bicubic_128)

    # Radial profile (average spectrum at each frequency)
    def radial_profile(spectrum):
        """Compute radially averaged power spectrum"""
        h, w = spectrum.shape
        center = (h // 2, w // 2)
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)

        # Bin by radius
        max_r = min(h, w) // 2
        radial_mean = np.zeros(max_r)
        for i in range(max_r):
            mask = (r == i)
            if mask.sum() > 0:
                radial_mean[i] = spectrum[mask].mean()

        return radial_mean

    profile_32 = radial_profile(spec_32)
    profile_64_model = radial_profile(spec_64_model)
    profile_128_model = radial_profile(spec_128_model)
    profile_64_bicubic = radial_profile(spec_64_bicubic)
    profile_128_bicubic = radial_profile(spec_128_bicubic)

    # Plot results
    fig = plt.figure(figsize=(18, 12))

    # Row 1: 2D Frequency Spectra
    ax1 = plt.subplot(3, 5, 1)
    ax1.imshow(np.log(spec_32 + 1), cmap='hot')
    ax1.set_title('32×32 (original)\nFrequency Spectrum')
    ax1.axis('off')

    ax2 = plt.subplot(3, 5, 2)
    ax2.imshow(np.log(spec_64_bicubic + 1), cmap='hot')
    ax2.set_title('64×64 Bicubic\nFrequency Spectrum')
    ax2.axis('off')

    ax3 = plt.subplot(3, 5, 3)
    ax3.imshow(np.log(spec_64_model + 1), cmap='hot')
    ax3.set_title('64×64 MAMBA-GINR\nFrequency Spectrum')
    ax3.axis('off')

    ax4 = plt.subplot(3, 5, 4)
    ax4.imshow(np.log(spec_128_bicubic + 1), cmap='hot')
    ax4.set_title('128×128 Bicubic\nFrequency Spectrum')
    ax4.axis('off')

    ax5 = plt.subplot(3, 5, 5)
    ax5.imshow(np.log(spec_128_model + 1), cmap='hot')
    ax5.set_title('128×128 MAMBA-GINR\nFrequency Spectrum')
    ax5.axis('off')

    # Row 2: Radial profiles (linear scale)
    ax6 = plt.subplot(3, 2, 3)
    freq_axis_32 = np.arange(len(profile_32))
    freq_axis_64 = np.arange(len(profile_64_model))
    freq_axis_128 = np.arange(len(profile_128_model))

    ax6.plot(freq_axis_32, profile_32, label='32×32 (original)', linewidth=2)
    ax6.plot(freq_axis_64, profile_64_bicubic[:len(freq_axis_64)],
             label='64×64 Bicubic', linestyle='--', alpha=0.7)
    ax6.plot(freq_axis_64, profile_64_model,
             label='64×64 MAMBA-GINR', linewidth=2)
    ax6.set_xlabel('Spatial Frequency (cycles)')
    ax6.set_ylabel('Power')
    ax6.set_title('Radial Power Spectrum (Linear Scale)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Row 2: Radial profiles (log scale)
    ax7 = plt.subplot(3, 2, 4)
    ax7.semilogy(freq_axis_32, profile_32 + 1e-10, label='32×32 (original)', linewidth=2)
    ax7.semilogy(freq_axis_128, profile_128_bicubic[:len(freq_axis_128)] + 1e-10,
                 label='128×128 Bicubic', linestyle='--', alpha=0.7)
    ax7.semilogy(freq_axis_128, profile_128_model + 1e-10,
                 label='128×128 MAMBA-GINR', linewidth=2)
    ax7.set_xlabel('Spatial Frequency (cycles)')
    ax7.set_ylabel('Power (log scale)')
    ax7.set_title('Radial Power Spectrum (Log Scale)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Nyquist frequencies
    nyquist_32 = 16  # 32/2
    nyquist_64 = 32  # 64/2
    nyquist_128 = 64  # 128/2

    ax6.axvline(nyquist_32, color='red', linestyle=':', label='Nyquist (32×32)')
    ax6.axvline(nyquist_64, color='orange', linestyle=':', label='Nyquist (64×64)')
    ax7.axvline(nyquist_64, color='orange', linestyle=':', label='Nyquist (64×64)')
    ax7.axvline(nyquist_128, color='purple', linestyle=':', label='Nyquist (128×128)')

    # Row 3: High-frequency analysis
    ax8 = plt.subplot(3, 2, 5)

    # Compare high-frequency content beyond original Nyquist
    hf_start = nyquist_32

    # For 64×64
    if len(profile_64_model) > hf_start:
        hf_model_64 = profile_64_model[hf_start:].sum()
        hf_bicubic_64 = profile_64_bicubic[hf_start:len(profile_64_model)].sum()

        ax8.bar(['Bicubic\n64×64', 'MAMBA-GINR\n64×64'],
                [hf_bicubic_64, hf_model_64],
                color=['gray', 'steelblue'])
        ax8.set_ylabel('Total High-Frequency Power\n(beyond 32×32 Nyquist)')
        ax8.set_title('High-Frequency Content (64×64)')
        ax8.grid(True, alpha=0.3, axis='y')

    ax9 = plt.subplot(3, 2, 6)

    # For 128×128
    if len(profile_128_model) > hf_start:
        hf_model_128 = profile_128_model[hf_start:].sum()
        hf_bicubic_128 = profile_128_bicubic[hf_start:len(profile_128_model)].sum()

        ax9.bar(['Bicubic\n128×128', 'MAMBA-GINR\n128×128'],
                [hf_bicubic_128, hf_model_128],
                color=['gray', 'steelblue'])
        ax9.set_ylabel('Total High-Frequency Power\n(beyond 32×32 Nyquist)')
        ax9.set_title('High-Frequency Content (128×128)')
        ax9.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('frequency_spectrum_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()

    # Print analysis
    print("="*70)
    print("FREQUENCY SPECTRUM ANALYSIS")
    print("="*70)
    print("\n📊 KEY OBSERVATIONS:\n")

    # Check if model generates new high frequencies
    ratio_64 = hf_model_64 / (hf_bicubic_64 + 1e-10)
    ratio_128 = hf_model_128 / (hf_bicubic_128 + 1e-10)

    print(f"High-frequency power ratio (Model / Bicubic):")
    print(f"  • 64×64:  {ratio_64:.3f}x")
    print(f"  • 128×128: {ratio_128:.3f}x")

    if ratio_64 < 1.5 and ratio_128 < 1.5:
        print("\n❌ VERDICT: Model is performing RESAMPLING, not super-resolution")
        print("   - High-frequency content similar to bicubic interpolation")
        print("   - No new details generated beyond training resolution")
        print("   - Frequency spectrum drops off at 32×32 Nyquist limit")
    elif ratio_64 > 2.0 or ratio_128 > 2.0:
        print("\n✅ VERDICT: Model is generating NEW high-frequency content")
        print("   - Significantly more high-freq power than bicubic")
        print("   - True super-resolution with hallucinated details")
    else:
        print("\n⚠️  VERDICT: Marginal high-frequency generation")
        print("   - Some new details, but limited")
        print("   - Between resampling and true super-resolution")

    print("\n" + "="*70)

    return {
        'spectra': {
            '32': spec_32,
            '64_model': spec_64_model,
            '64_bicubic': spec_64_bicubic,
            '128_model': spec_128_model,
            '128_bicubic': spec_128_bicubic
        },
        'profiles': {
            '32': profile_32,
            '64_model': profile_64_model,
            '64_bicubic': profile_64_bicubic,
            '128_model': profile_128_model,
            '128_bicubic': profile_128_bicubic
        },
        'metrics': {
            'hf_ratio_64': ratio_64,
            'hf_ratio_128': ratio_128
        }
    }


# Run analysis
print("Analyzing frequency content of super-resolved images...")
freq_analysis = analyze_frequency_spectrum(model, test_images, device)
