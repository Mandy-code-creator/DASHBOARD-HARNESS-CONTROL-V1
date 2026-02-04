# ---------- DISTRIBUTION : LAB + LINE COMBINED ----------

lab = sub["Hardness_LAB"].dropna()
line = sub["Hardness_LINE"].dropna()

if len(lab) >= 10 and len(line) >= 10:

    mean_lab, std_lab = lab.mean(), lab.std(ddof=1)
    mean_line, std_line = line.mean(), line.std(ddof=1)

    # ===== 3 SIGMA RANGE (CHUNG)
    x_min = min(mean_lab - 3*std_lab, mean_line - 3*std_line)
    x_max = max(mean_lab + 3*std_lab, mean_line + 3*std_line)

    bins = np.linspace(x_min, x_max, 25)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # ===== HISTOGRAM
    ax.hist(lab, bins=bins, density=True,
            alpha=0.35, edgecolor="black",
            label="LAB")

    ax.hist(line, bins=bins, density=True,
            alpha=0.35, edgecolor="black",
            label="LINE")

    # ===== NORMAL CURVE (±3σ)
    xs = np.linspace(x_min, x_max, 400)

    ys_lab = (1/(std_lab*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_lab)/std_lab)**2)
    ys_line = (1/(std_line*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mean_line)/std_line)**2)

    ax.plot(xs, ys_lab, linewidth=2.5, label="LAB Normal (±3σ)")
    ax.plot(xs, ys_line, linewidth=2.5, linestyle="--", label="LINE Normal (±3σ)")

    # ===== SPEC LIMIT
    ax.axvline(lo, linestyle="--", linewidth=2, label=f"LSL = {lo}")
    ax.axvline(hi, linestyle="--", linewidth=2, label=f"USL = {hi}")

    # ===== MEAN
    ax.axvline(mean_lab, linestyle=":", linewidth=2, label=f"LAB Mean {mean_lab:.2f}")
    ax.axvline(mean_line, linestyle=":", linewidth=2, label=f"LINE Mean {mean_line:.2f}")

    ax.set_title("Hardness Distribution – LAB vs LINE (3σ)", weight="bold")
    ax.set_xlabel("Hardness (HRB)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    # ===== NOTE (BÊN NGOÀI)
    note = (
        f"LAB:\n"
        f"  N={len(lab)}\n"
        f"  Mean={mean_lab:.2f}\n"
        f"  Std={std_lab:.2f}\n\n"
        f"LINE:\n"
        f"  N={len(line)}\n"
        f"  Mean={mean_line:.2f}\n"
        f"  Std={std_line:.2f}"
    )

    ax.text(
        1.02, 0.5,
        note,
        transform=ax.transAxes,
        va="center",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.85), frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
