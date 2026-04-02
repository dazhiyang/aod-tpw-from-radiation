#################################################################################
# 12.test_analysis.R
# Composite test figure:
#   top-left  : FGE violin (MERRA-2, TabPFN; plus libRadtran OE if TEST_OE exists)
#   top-right : SHAP summary-style panel (AOD550 + alpha)
#   bottom    : irradiance scatter (MERRA-2, TabPFN, AERONET)
#
# Theme follows 11.train_analysis.R (plot.size=9, serif, 160 mm width).
#################################################################################

rm(list = ls(all = TRUE))
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(scattermore)
})

# Used inside ggplot2::aes() / dplyr pipelines; avoids R CMD check / lintr NSE notes.
utils::globalVariables(c("shap_value", "shap_plot", "feature_lab", "feature_value_scaled", "feature_value", "target"))

.get.script.dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args)
  if (length(m) > 0) return(dirname(normalizePath(sub("^--file=", "", args[m[1]]))))
  if (!is.null(sys.frame(1)$ofile)) return(dirname(normalizePath(sys.frame(1)$ofile)))
  return(getwd())
}
dir0 <- normalizePath(file.path(.get.script.dir(), ".."), mustWork = FALSE)

stn <- Sys.getenv("STATION", "PAL")
yr  <- as.integer(Sys.getenv("YEAR", "2024"))
lhs.n <- as.integer(Sys.getenv("LHS_N", "500"))
k.suffix <- ifelse(lhs.n == 500, "_0.5k", ifelse(lhs.n >= 1000, paste0("_", lhs.n/1000, "k"), paste0("_", lhs.n)))

pred.oe.path <- Sys.getenv("TEST_COMBINED", file.path(dir0, "Data", paste0(stn, "_", yr, "_pred_oe", k.suffix, ".txt")))
test.oe.path <- Sys.getenv("TEST_OE", file.path(dir0, "Data", paste0(stn, "_", yr, "_test_oe.txt")))
train.oe.path <- Sys.getenv("TRAIN_OE", file.path(dir0, "Data", paste0(stn, "_", yr, "_train_oe", k.suffix, ".txt")))
irr.path <- Sys.getenv("IRRADIANCE_IN", file.path(dir0, "Data", paste0(stn, "_", yr, "_test_irradiance", k.suffix, ".txt")))
shap.a.path <- Sys.getenv("SHAP_ALPHA", file.path(dir0, "Data", paste0(stn, "_", yr, "_shap_oe_alpha", k.suffix, ".txt")))
shap.b.path <- Sys.getenv("SHAP_BETA", file.path(dir0, "Data", paste0(stn, "_", yr, "_shap_oe_beta", k.suffix, ".txt")))
out.fig <- Sys.getenv("OUTPUT_FIG", file.path(dir0, "tex", "test_results.pdf"))

plot.size <- 9
line.size <- 0.3
point.size <- 0.9
fig.w.mm <- as.numeric(Sys.getenv("FIG_W_MM", "160"))
fig.h.mm <- as.numeric(Sys.getenv("FIG_H_MM", "160"))
aod.shap.xlim <- c(-0.52, 0.52)

lam550 <- 0.55
lambda.ref <- as.numeric(Sys.getenv("ANGSTROM_BETA_REF_UM", "1.0"))
aod550 <- function(beta, alpha) beta * (lam550 / lambda.ref)^(-alpha)

col.merra <- "#E69F00"
col.sky <- "#56B4E9"     # Wong sky blue — BNI in (c) scatter + SHAP colour bar (unchanged)
col.tabpfn <- "#CC79A7"  # Wong reddish purple — TabPFN in (a) violin only
col.librt <- "#009E73"   # Wong bluish green — libRadtran OE in (a) violin
col.dhi <- "#009E73"     # DHI in (c) scatter
src.colors <- c("MERRA-2" = col.merra, "TabPFN" = col.tabpfn, "AERONET" = "black")
comp.colors <- c("GHI" = col.merra, "BNI" = col.sky, "DHI" = col.dhi)
shap.cmap <- c(col.sky, "#F2F2F2", col.merra)

theme.common <- theme_bw(base_size = plot.size, base_family = "serif") +
  theme(
    text = element_text(family = "serif", size = plot.size),
    axis.text = element_text(size = plot.size),
    axis.title = element_text(size = plot.size),
    plot.title = element_text(size = plot.size),
    legend.text = element_text(size = plot.size),
    legend.title = element_blank(),
    strip.text = element_text(size = plot.size, margin = margin(0.3, 1, 0.3, 1, "pt")),
    strip.background = element_rect(fill = "white", colour = "black", linewidth = line.size),
    panel.grid.major = element_line(colour = alpha("black", 0.35), linewidth = line.size),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = NA, colour = "black", linewidth = line.size),
    axis.line = element_blank(),
    axis.ticks = element_line(colour = "black", linewidth = line.size),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.margin = margin(0, 0, 0, 0),
    legend.key = element_rect(fill = NA, colour = NA),
    legend.key.size = unit(plot.size * 1.2, "pt"),
    legend.background = element_blank(),
    panel.spacing.x = unit(0.012, "npc"),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

fge.vec <- function(pred, ref) {
  den <- pred + ref
  out <- rep(NA_real_, length(pred))
  ok <- is.finite(pred) & is.finite(ref) & is.finite(den) & den > 0
  out[ok] <- 2 * abs(pred[ok] - ref[ok]) / den[ok]
  out
}
metrics <- function(pred, ref) {
  ok <- is.finite(pred) & is.finite(ref)
  p <- pred[ok]; r <- ref[ok]
  if (length(p) == 0) return(data.frame(n=0, mbe=NA, rmse=NA, fb=NA, fge=NA))
  den.fb <- mean(p) + mean(r)
  fge <- fge.vec(p, r)
  data.frame(
    n = length(p),
    mbe = mean(p - r),
    rmse = sqrt(mean((p - r)^2)),
    fb = ifelse(den.fb > 0, 2 * (mean(r) - mean(p)) / den.fb, NA_real_),
    fge = mean(fge, na.rm = TRUE)
  )
}

stopifnot(file.exists(pred.oe.path), file.exists(train.oe.path), file.exists(irr.path), file.exists(shap.a.path), file.exists(shap.b.path))
te <- read.delim(pred.oe.path, sep = "\t", comment.char = "#")
tr <- read.delim(train.oe.path, sep = "\t", comment.char = "#")
irr <- read.delim(irr.path, sep = "\t", comment.char = "#")
shap.a <- read.delim(shap.a.path, sep = "\t", comment.char = "#")
shap.b <- read.delim(shap.b.path, sep = "\t", comment.char = "#")

has.test.oe <- file.exists(test.oe.path)
if (has.test.oe) {
  ts <- read.delim(test.oe.path, sep = "\t", comment.char = "#")
  te <- dplyr::inner_join(
    te,
    dplyr::select(ts, time_utc, beta_oe, alpha_oe),
    by = "time_utc"
  )
  if (nrow(te) == 0) stop("No overlapping time_utc between pred OE and TEST_OE")
  te$aod_phys <- aod550(te$beta_oe, te$alpha_oe)
} else {
  message("Note: TEST_OE not found (", test.oe.path, "); violin shows MERRA-2 and TabPFN only.")
}

te$aod_oe <- aod550(te$beta_pred_oe, te$alpha_pred_oe)
te$aod_merra <- aod550(te$merra_BETA, te$merra_ALPHA)

ds.levels <- if (has.test.oe) c("MERRA-2", "TabPFN", "libRadtran OE") else c("MERRA-2", "TabPFN")
fge.parts <- list(
  data.frame(dataset = "MERRA-2", variable = "AOD[550]", value = fge.vec(te$aod_merra, te$aeronet_aod550)),
  data.frame(dataset = "TabPFN", variable = "AOD[550]", value = fge.vec(te$aod_oe, te$aeronet_aod550)),
  data.frame(dataset = "MERRA-2", variable = "Angstrom~alpha", value = fge.vec(te$merra_ALPHA, te$aeronet_alpha)),
  data.frame(dataset = "TabPFN", variable = "Angstrom~alpha", value = fge.vec(te$alpha_pred_oe, te$aeronet_alpha))
)
if (has.test.oe) {
  fge.parts <- c(
    fge.parts,
    list(
      data.frame(dataset = "libRadtran OE", variable = "AOD[550]", value = fge.vec(te$aod_phys, te$aeronet_aod550)),
      data.frame(dataset = "libRadtran OE", variable = "Angstrom~alpha", value = fge.vec(te$alpha_oe, te$aeronet_alpha))
    )
  )
}
fge.df <- dplyr::bind_rows(fge.parts) %>% filter(is.finite(value))
fge.df$dataset <- factor(fge.df$dataset, levels = ds.levels)
fge.df$variable <- factor(fge.df$variable, levels = c("AOD[550]", "Angstrom~alpha"))

ann.parts <- list(
  metrics(te$aod_merra, te$aeronet_aod550) %>% mutate(dataset = "MERRA-2", variable = "AOD[550]"),
  metrics(te$aod_oe, te$aeronet_aod550) %>% mutate(dataset = "TabPFN", variable = "AOD[550]"),
  metrics(te$merra_ALPHA, te$aeronet_alpha) %>% mutate(dataset = "MERRA-2", variable = "Angstrom~alpha"),
  metrics(te$alpha_pred_oe, te$aeronet_alpha) %>% mutate(dataset = "TabPFN", variable = "Angstrom~alpha")
)
if (has.test.oe) {
  ann.parts <- c(
    ann.parts,
    list(
      metrics(te$aod_phys, te$aeronet_aod550) %>% mutate(dataset = "libRadtran OE", variable = "AOD[550]"),
      metrics(te$alpha_oe, te$aeronet_alpha) %>% mutate(dataset = "libRadtran OE", variable = "Angstrom~alpha")
    )
  )
}
ann <- dplyr::bind_rows(ann.parts)
ann$label <- sprintf("MBE=%.4f\nRMSE=%.4f\nFB=%.4f\nFGE=%.4f", ann$mbe, ann$rmse, ann$fb, ann$fge)
ann$y <- as.numeric(Sys.getenv("ANNOT_Y", "1.5"))
ann$dataset <- factor(ann$dataset, levels = ds.levels)
ann$variable <- factor(ann$variable, levels = c("AOD[550]", "Angstrom~alpha"))
ann$xpos <- as.numeric(ann$dataset) - 0.42
ann$hjust <- 0

fill.cols <- c("MERRA-2" = col.merra, "TabPFN" = col.tabpfn)
if (has.test.oe) fill.cols <- c(fill.cols, "libRadtran OE" = col.librt)

p.violin <- ggplot(fge.df, aes(x = dataset, y = value, fill = dataset, colour = dataset)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.35, colour = "#4d4d4d") +
  geom_violin(alpha = 0.30, trim = FALSE, linewidth = 0.25) +
  geom_boxplot(width = 0.12, outlier.alpha = 0.15, fill = "white", linewidth = 0.25) +
  geom_text(data = ann, aes(x = xpos, y = y, label = label), inherit.aes = FALSE,
            hjust = 0,
            vjust = 1, lineheight = 0.9, size = plot.size / .pt, colour = "black") +
  facet_wrap(~ variable, nrow = 2, ncol = 1, scales = "free_y", labeller = label_parsed) +
  scale_fill_manual(values = fill.cols) +
  scale_colour_manual(values = fill.cols) +
  labs(title = "(a) FGE", x = "Dataset", y = "FGE") +
  theme.common +
  theme(
    legend.position = "none",
    panel.grid.major.y = element_blank()
  )

feature.labels <- c(
  ghi = "italic(G)[italic(h)]",
  bni = "italic(B)[italic(n)]",
  dhi = "italic(D)[italic(h)]",
  zenith = "theta[italic(z)]",
  merra_ALPHA = "alpha[merra2]",
  merra_BETA = "beta[merra2]",
  merra_ALBEDO = "rho",
  merra_TO3 = "italic(u)[italic(o)]",
  merra_PS = "italic(p)[italic(s)]",
  merra_TQV = "italic(w)"
)

shap.long <- bind_rows(
  shap.b %>% mutate(target = "AOD[550]"),
  shap.a %>% mutate(target = "Angstrom~alpha")
) %>%
  filter(!(target == "Angstrom~alpha" & sample_index == 64)) %>%
  mutate(
    feature_lab = ifelse(feature %in% names(feature.labels), feature.labels[feature], feature),
    target = factor(target, levels = c("AOD[550]", "Angstrom~alpha"))
  )
# Rank SHAP y-axis by mean |SHAP| for AOD (beta) only; same order on both panels.
rank.tbl <- shap.long %>%
  filter(target == "AOD[550]") %>%
  group_by(feature_lab) %>%
  summarise(mabs = mean(abs(shap_value), na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mabs))
miss.feats <- setdiff(unique(as.character(shap.long$feature_lab)), rank.tbl$feature_lab)
if (length(miss.feats) > 0) {
  rank.tbl <- bind_rows(rank.tbl, data.frame(feature_lab = miss.feats, mabs = 0, stringsAsFactors = FALSE))
}
shap.long$feature_lab <- factor(shap.long$feature_lab, levels = rev(rank.tbl$feature_lab))
# SHAP-style color normalization: normalize feature values within each feature.
shap.long <- shap.long %>%
  group_by(feature_lab) %>%
  mutate(feature_value_scaled = scales::rescale(feature_value, to = c(0, 1), na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(
    shap_plot = ifelse(
      as.character(target) == "AOD[550]",
      pmin(pmax(shap_value, aod.shap.xlim[1]), aod.shap.xlim[2]),
      shap_value
    )
  )

# One ggplot, two facets (AOD | alpha). AOD x is squished into aod.shap.xlim (ggplot2 has no per-facet coord_cartesian).
p.shap <- ggplot(shap.long, aes(x = shap_plot, y = feature_lab, colour = feature_value_scaled)) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = line.size, colour = "#737373") +
  geom_point(alpha = 0.70, size = point.size, position = position_jitter(height = 0.18, width = 0)) +
  facet_wrap(~ target, nrow = 1, scales = "free_x", labeller = label_parsed) +
  scale_y_discrete(labels = function(x) parse(text = x)) +
  scale_colour_gradientn(
    colours = shap.cmap,
    limits = c(0, 1),
    guide = guide_colourbar(
      direction = "horizontal",
      barwidth = unit(plot.size * 10, "pt"),
      barheight = unit(plot.size * 0.45, "pt")
    )
  ) +
  labs(title = "(b) SHAP", x = "SHAP value", y = "Feature") +
  theme.common +
  theme(legend.position = "bottom")

irr.long <- bind_rows(
  irr %>% transmute(panel = "MERRA-2", component = "GHI", measured = ghi, forward = ghi_merra),
  irr %>% transmute(panel = "MERRA-2", component = "BNI", measured = bni, forward = bni_merra),
  irr %>% transmute(panel = "MERRA-2", component = "DHI", measured = dhi, forward = dhi_merra),
  irr %>% transmute(panel = "TabPFN", component = "GHI", measured = ghi, forward = ghi_oe),
  irr %>% transmute(panel = "TabPFN", component = "BNI", measured = bni, forward = bni_oe),
  irr %>% transmute(panel = "TabPFN", component = "DHI", measured = dhi, forward = dhi_oe),
  irr %>% transmute(panel = "AERONET", component = "GHI", measured = ghi, forward = ghi_aeronet),
  irr %>% transmute(panel = "AERONET", component = "BNI", measured = bni, forward = bni_aeronet),
  irr %>% transmute(panel = "AERONET", component = "DHI", measured = dhi, forward = dhi_aeronet)
) %>% filter(is.finite(measured), is.finite(forward))
irr.long$panel <- factor(irr.long$panel, levels = c("MERRA-2", "TabPFN", "AERONET"))
irr.long$component <- factor(irr.long$component, levels = c("GHI", "BNI", "DHI"))

for (pnl in levels(irr.long$panel)) {
  for (cmp in levels(irr.long$component)) {
    s <- irr.long %>% filter(panel == pnl, component == cmp)
    e <- s$forward - s$measured
    cat(sprintf("%s %s: MBE=%.3f W m^-2, RMSE=%.3f W m^-2\n",
                pnl, cmp, mean(e), sqrt(mean(e^2))))
  }
}

lo <- min(irr.long$measured); hi <- max(irr.long$measured)
line.df <- data.frame(panel = rep(levels(irr.long$panel), each = 2),
                      measured = rep(c(lo, hi), times = length(levels(irr.long$panel))),
                      forward = rep(c(lo, hi), times = length(levels(irr.long$panel))),
                      grp = 0)
# Add pooled per-panel annotation (MBE / RMSE% / R² style).
irr.stats <- irr.long %>%
  group_by(panel) %>%
  summarise(
    rmse = sqrt(mean((forward - measured)^2, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    tx = lo + 0.04 * (hi - lo),
    ty = lo + 0.95 * (hi - lo),
    txt = sprintf("'RMSE ='~%.1f~W~m^{-2}", rmse)
  )

p.irr <- ggplot(irr.long, aes(x = measured, y = forward, colour = component)) +
  geom_scattermore(
    pointsize = 3,
    pixels = c(512L, 512L),
    interpolate = TRUE
  ) +
  geom_line(data = line.df, aes(x = measured, y = forward, group = grp), inherit.aes = FALSE,
            linetype = "dashed", linewidth = line.size, colour = "black", alpha = 0.65) +
  geom_text(data = irr.stats, aes(x = tx, y = ty, label = txt), inherit.aes = FALSE,
            hjust = 0, vjust = 1, size = plot.size / .pt, colour = "black", lineheight = 0.9, parse = TRUE) +
  facet_wrap(~ panel, nrow = 1, scales = "fixed") +
  scale_colour_manual(values = comp.colors) +
  coord_fixed(ratio = 1, xlim = c(lo, hi), ylim = c(lo, hi), expand = FALSE) +
  labs(title = "(c) Forward irradiance",
       x = expression("Measured ("*W~m^{-2}*")"),
       y = expression("libRadtran forward ("*W~m^{-2}*")")) +
  theme.common +
  theme(
    legend.position = "bottom",
    legend.box.spacing = unit(0, "pt"),
    legend.spacing.y = unit(0, "pt"),
    legend.margin = margin(-1, 0, -1, 0, "pt"),
    legend.key.height = unit(plot.size * 0.85, "pt"),
    legend.key.width = unit(plot.size * 1.0, "pt"),
    axis.title.x = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt")),
    plot.margin = margin(0, 2, 0, 2, "pt")
  )

layout.design <- "
AB
CC
"
all.fig <- (
  p.violin + p.shap + p.irr +
    plot_layout(
      design = layout.design,
      widths = c(1, 1),
      heights = c(1.5, 1)
    )
) & theme(plot.margin = margin(2, 2, 2, 2, "pt"))

dir.create(dirname(out.fig), recursive = TRUE, showWarnings = FALSE)
ggsave(out.fig, all.fig, width = fig.w.mm, height = fig.h.mm, units = "mm", dpi = 300)
cat(sprintf("Wrote: %s\n", out.fig))
