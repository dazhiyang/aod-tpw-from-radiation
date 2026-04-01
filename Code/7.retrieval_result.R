#################################################################################
# This code is written by Dazhi Yang
# Department of Electrical Engineering and Automation, Harbin Institute of Technology
# email: yangdazhi.nus@gmail.com
#################################################################################
# 7.retrieval_result.R
# Densities of AOD_550 and alpha, then scatter vs AERONET.
# TabPFN mode (USE_TABPFN=TRUE): merges pred_ls + pred_oe from step 5.
# Retrieval mode (default): merges LHS + train_ls + train_oe from steps 3-4.
#
# Usage:
#   Rscript Code/7.retrieval_result.R
#   USE_TABPFN=TRUE Rscript Code/7.retrieval_result.R
#################################################################################

#Clear all workspace
rm(list = ls(all = TRUE))
libs <- c("dplyr", "tidyr", "ggplot2", "gridExtra", "grid", "patchwork")
invisible(lapply(libs, library, character.only = TRUE))

#################################################################################
# Inputs
#################################################################################
# resolve project root: works with Rscript and source()
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
use.tabpfn <- tolower(Sys.getenv("USE_TABPFN", "")) %in% c("1", "true", "yes")

plot.size <- 9; line.size <- 0.3; point.size <- 1.5
lam550 <- 0.55; lambda.ref <- 1.0
aod.x.max <- 0.6; density.n <- 512
ribbon.alpha <- 0.18; density.lw <- 0.12; density.aeronet.lw <- 0.55

# Wong colorblind-friendly palette
col.merra   <- "#E69F00"  # orange
col.aeronet <- "#000000"  # black
col.ls      <- "#56B4E9"  # sky blue
col.oe      <- "#009E73"  # bluish green

# source labels
src.merra   <- "MERRA-2"
src.aeronet <- "AERONET"
src.ls      <- ifelse(use.tabpfn, "TabPFN LS", "LS retrieval")
src.oe      <- ifelse(use.tabpfn, "TabPFN OE", "OE retrieval")
src.order   <- c(src.aeronet, src.merra, src.ls, src.oe)
src.colors  <- c(col.aeronet, col.merra, col.ls, col.oe)
names(src.colors) <- src.order
res.order   <- c(src.merra, src.ls, src.oe)
res.colors  <- c(col.merra, col.ls, col.oe)
names(res.colors) <- res.order

# file paths
pred.ls.path <- file.path(dir0, "Data", paste0(stn, "_", yr, "_pred_ls", k.suffix, ".txt"))
pred.oe.path <- file.path(dir0, "Data", paste0(stn, "_", yr, "_pred_oe", k.suffix, ".txt"))
lhs.path     <- file.path(dir0, "Data", paste0(stn, "_", yr, "_train",    k.suffix, ".txt"))
ret.ls.path  <- file.path(dir0, "Data", paste0(stn, "_", yr, "_train_ls", k.suffix, ".txt"))
ret.oe.path  <- file.path(dir0, "Data", paste0(stn, "_", yr, "_train_oe", k.suffix, ".txt"))

fig.name <- ifelse(use.tabpfn, "train_results_tabpfn.pdf", "train_results.pdf")
fig.path <- Sys.getenv("OUTPUT_FIG", file.path(dir0, "tex", "figures", fig.name))

fig.w.mm <- 160; fig.h.density.mm <- 45; fig.h.scatter.mm <- 62
#################################################################################

#################################################################################
# Helper functions
#################################################################################
aod550 <- function(beta, alpha) beta * (lam550 / lambda.ref)^(-alpha)

metrics.vs.ref <- function(obs, mod) {
  ok <- is.finite(obs) & is.finite(mod)
  obs <- obs[ok]; mod <- mod[ok]; n <- length(obs)
  if (n == 0) return(data.frame(n=0, mbe=NA, rmse=NA, fb=NA, fge=NA))
  mo <- mean(obs); mm <- mean(mod); denom <- mo + mm
  data.frame(
    n    = n,
    mbe  = mean(mod - obs),
    rmse = sqrt(mean((mod - obs)^2)),
    fb   = ifelse(denom > 0, 2 * (mo - mm) / denom, NA),
    fge  = ifelse(denom > 0, 2 * mean(abs(obs - mod)) / denom, NA)
  )
}

metrics.text <- function(m) {
  paste0("n = ", m$n, "\n",
         "MBE = ", formatC(m$mbe, format="f", digits=4), "\n",
         "RMSE = ", formatC(m$rmse, format="f", digits=4), "\n",
         "FB = ", formatC(m$fb, format="f", digits=4), "\n",
         "FGE = ", formatC(m$fge, format="f", digits=4))
}

#################################################################################
# Data loading and merge
#################################################################################
if (use.tabpfn) {
  stopifnot(file.exists(pred.ls.path), file.exists(pred.oe.path))
  pls <- read.delim(pred.ls.path, sep = "\t", comment.char = "#")
  poe <- read.delim(pred.oe.path, sep = "\t", comment.char = "#")
  df <- merge(
    pls[, c("time_utc", "merra_ALPHA", "merra_BETA", "aeronet_aod550", "aeronet_alpha", "beta_pred_ls", "alpha_pred_ls")],
    poe[, c("time_utc", "beta_pred_oe", "alpha_pred_oe")],
    by = "time_utc"
  )
  b.ls <- "beta_pred_ls"; a.ls <- "alpha_pred_ls"
  b.oe <- "beta_pred_oe"; a.oe <- "alpha_pred_oe"
  mode.tag <- "TabPFN pred_ls + pred_oe"
} else {
  stopifnot(file.exists(lhs.path), file.exists(ret.ls.path), file.exists(ret.oe.path))
  lhs <- read.delim(lhs.path,    sep = "\t", comment.char = "#")
  rls <- read.delim(ret.ls.path, sep = "\t", comment.char = "#")
  roe <- read.delim(ret.oe.path, sep = "\t", comment.char = "#")
  df <- merge(
    lhs[, c("time_utc", "merra_ALPHA", "merra_BETA", "aeronet_aod550", "aeronet_alpha")],
    rls[, c("time_utc", "beta_ls", "alpha_ls")],
    by = "time_utc"
  )
  df <- merge(df, roe[, c("time_utc", "beta_oe", "alpha_oe")], by = "time_utc")
  b.ls <- "beta_ls"; a.ls <- "alpha_ls"
  b.oe <- "beta_oe"; a.oe <- "alpha_oe"
  mode.tag <- "LS + OE retrieval"
}

# drop incomplete rows
df <- df[complete.cases(df[, c("merra_ALPHA", "merra_BETA", "aeronet_aod550", "aeronet_alpha", b.ls, a.ls, b.oe, a.oe)]), ]
stopifnot(nrow(df) > 0)
cat(sprintf("Merged: %d rows (%s).\n", nrow(df), mode.tag))

#################################################################################
# Compute AOD_550 and alpha arrays
#################################################################################
aod.merra   <- aod550(df$merra_BETA,    df$merra_ALPHA)
aod.aeronet <- df$aeronet_aod550
aod.ls      <- aod550(df[[b.ls]], df[[a.ls]])
aod.oe      <- aod550(df[[b.oe]], df[[a.oe]])

alpha.merra   <- df$merra_ALPHA
alpha.aeronet <- df$aeronet_alpha
alpha.ls      <- df[[a.ls]]
alpha.oe      <- df[[a.oe]]

#################################################################################
# Metrics
#################################################################################
met.aod.m  <- metrics.vs.ref(aod.aeronet, aod.merra)
met.aod.ls <- metrics.vs.ref(aod.aeronet, aod.ls)
met.aod.oe <- metrics.vs.ref(aod.aeronet, aod.oe)
met.a.m    <- metrics.vs.ref(alpha.aeronet, alpha.merra)
met.a.ls   <- metrics.vs.ref(alpha.aeronet, alpha.ls)
met.a.oe   <- metrics.vs.ref(alpha.aeronet, alpha.oe)

ls.tag <- ifelse(use.tabpfn, "TabPFN LS", "LS")
oe.tag <- ifelse(use.tabpfn, "TabPFN OE", "OE")
for (info in list(
  c("AOD MERRA", met.aod.m), c(paste("AOD", ls.tag), met.aod.ls), c(paste("AOD", oe.tag), met.aod.oe),
  c("Alpha MERRA", met.a.m), c(paste("Alpha", ls.tag), met.a.ls), c(paste("Alpha", oe.tag), met.a.oe)
)) {
  m <- info[-1]
  cat(sprintf("%s vs AERONET: n=%d, MBE=%.6f, RMSE=%.6f, FB=%.6f, FGE=%.6f\n",
              info[1], as.integer(m$n), m$mbe, m$rmse, m$fb, m$fge))
}

#################################################################################
# Common ggplot theme (matches Python version)
#################################################################################
theme.common <- theme_bw(base_size = plot.size, base_family = "serif") +
  theme(
    text             = element_text(family = "serif", size = plot.size),
    axis.text        = element_text(size = plot.size),
    axis.title       = element_text(size = plot.size),
    plot.title       = element_text(size = plot.size),
    legend.text      = element_text(size = plot.size),
    legend.title     = element_blank(),
    strip.text       = element_text(size = plot.size, margin = margin(0.3, 1, 0.3, 1, "pt")),
    strip.background = element_rect(fill = "white", colour = "black", linewidth = line.size),
    panel.grid.major = element_line(colour = alpha("black", 0.35), linewidth = line.size),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(fill = NA, colour = "black", linewidth = line.size),
    axis.line        = element_blank(),
    axis.ticks       = element_line(colour = "black", linewidth = line.size),
    legend.position  = "bottom",
    legend.direction = "horizontal",
    legend.margin    = margin(0, 0, 0, 0),
    legend.key       = element_rect(fill = NA, colour = NA),
    legend.key.size  = unit(plot.size * 1.2, "pt"),
    legend.background = element_blank(),
    panel.spacing.x  = unit(0.012, "npc"),
    plot.background  = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

#################################################################################
# Panel 1: density of AOD_550 and alpha
#################################################################################
long.aod <- rbind(
  data.frame(panel = "AOD[550]", source = src.merra,   value = aod.merra[is.finite(aod.merra) & aod.merra <= aod.x.max]),
  data.frame(panel = "AOD[550]", source = src.aeronet, value = aod.aeronet[is.finite(aod.aeronet) & aod.aeronet <= aod.x.max]),
  data.frame(panel = "AOD[550]", source = src.ls,      value = aod.ls[is.finite(aod.ls) & aod.ls <= aod.x.max]),
  data.frame(panel = "AOD[550]", source = src.oe,      value = aod.oe[is.finite(aod.oe) & aod.oe <= aod.x.max])
)
long.alpha <- rbind(
  data.frame(panel = "Angstrom~alpha", source = src.merra,   value = alpha.merra[is.finite(alpha.merra)]),
  data.frame(panel = "Angstrom~alpha", source = src.aeronet, value = alpha.aeronet[is.finite(alpha.aeronet)]),
  data.frame(panel = "Angstrom~alpha", source = src.ls,      value = alpha.ls[is.finite(alpha.ls)]),
  data.frame(panel = "Angstrom~alpha", source = src.oe,      value = alpha.oe[is.finite(alpha.oe)])
)
long.df <- rbind(long.aod, long.alpha)
long.df$panel  <- factor(long.df$panel, levels = c("AOD[550]", "Angstrom~alpha"))
long.df$source <- factor(long.df$source, levels = src.order)

# separate AERONET (line only) from ribbon sources
long.ribbon  <- long.df[long.df$source != src.aeronet, ]
long.aeronet <- long.df[long.df$source == src.aeronet, ]

p.density <- ggplot() +
  stat_density(data = long.ribbon, aes(x = value, fill = source, colour = source),
               geom = "area", position = "identity", alpha = ribbon.alpha,
               linewidth = density.lw, n = density.n) +
  stat_density(data = long.aeronet, aes(x = value, colour = source),
               geom = "line", position = "identity",
               linewidth = density.aeronet.lw, n = density.n) +
  facet_wrap(~ panel, nrow = 1, scales = "free", labeller = label_parsed) +
  scale_fill_manual(values = src.colors, breaks = src.order, guide = "none") +
  scale_colour_manual(values = src.colors, breaks = src.order,
                      guide = guide_legend(override.aes = list(
                        fill      = c(NA, col.merra, col.ls, col.oe),
                        alpha     = c(1, ribbon.alpha, ribbon.alpha, ribbon.alpha),
                        linewidth = c(density.aeronet.lw, density.lw, density.lw, density.lw)
                      ))) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  labs(x = "", y = "Density", title = expression(paste("(a) Density of ", AOD[550], " and ", alpha))) +
  theme.common +
  theme(legend.box.spacing = unit(0, "pt"),
        legend.spacing.y   = unit(0, "pt"),
        plot.margin        = margin(2, 2, 0, 2, "pt"))

#################################################################################
# Panel 2: AOD_550 scatter (AERONET x vs model y), three facets
#################################################################################
n.row <- nrow(df)
aod.scatter <- data.frame(
  aeronet = rep(aod.aeronet, 3),
  model   = c(aod.merra, aod.ls, aod.oe),
  source  = factor(rep(res.order, each = n.row), levels = res.order)
)
hi.aod <- 0.76
aod.label <- data.frame(
  source  = factor(res.order, levels = res.order),
  tx      = rep(hi.aod * 0.04, 3),
  ty      = rep(hi.aod * 0.95, 3),
  txt     = c(metrics.text(met.aod.m), metrics.text(met.aod.ls), metrics.text(met.aod.oe))
)

p.aod <- ggplot(aod.scatter, aes(x = aeronet, y = model, colour = source)) +
  geom_point(alpha = 0.45, size = point.size, stroke = 0) +
  geom_abline(intercept = 0, slope = 1, colour = "#737373", linetype = "dashed", linewidth = line.size) +
  geom_text(data = aod.label, aes(x = tx, y = ty, label = txt),
            hjust = 0, vjust = 1, size = plot.size / .pt, colour = "black", lineheight = 0.9) +
  facet_wrap(~ source, nrow = 1) +
  scale_x_continuous(limits = c(0, 0.76), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 0.76), expand = c(0, 0)) +
  scale_colour_manual(values = res.colors) +
  labs(x = "AERONET (reference)", y = "Model", title = expression(paste("(b) ", AOD[550]))) +
  theme.common + theme(legend.position = "none", panel.spacing.x = unit(0.01, "npc"),
                        plot.margin = margin(2, 2, 0, 2, "pt"))

#################################################################################
# Panel 3: alpha scatter (AERONET x vs model y), three facets
#################################################################################
alpha.scatter <- data.frame(
  aeronet = rep(alpha.aeronet, 3),
  model   = c(alpha.merra, alpha.ls, alpha.oe),
  source  = factor(rep(res.order, each = n.row), levels = res.order)
)
hi.alpha <- max(max(alpha.aeronet, alpha.merra, alpha.ls, alpha.oe, na.rm = TRUE) * 1.05, 0.5)
alpha.label <- data.frame(
  source  = factor(res.order, levels = res.order),
  tx      = rep(hi.alpha * 0.04, 3),
  ty      = rep(hi.alpha * 0.95, 3),
  txt     = c(metrics.text(met.a.m), metrics.text(met.a.ls), metrics.text(met.a.oe))
)

p.alpha <- ggplot(alpha.scatter, aes(x = aeronet, y = model, colour = source)) +
  geom_point(alpha = 0.45, size = point.size, stroke = 0) +
  geom_abline(intercept = 0, slope = 1, colour = "#737373", linetype = "dashed", linewidth = line.size) +
  geom_text(data = alpha.label, aes(x = tx, y = ty, label = txt),
            hjust = 0, vjust = 1, size = plot.size / .pt, colour = "black", lineheight = 0.9) +
  facet_wrap(~ source, nrow = 1) +
  scale_x_continuous(limits = c(0, hi.alpha), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, hi.alpha), expand = c(0, 0)) +
  scale_colour_manual(values = res.colors) +
  labs(x = "AERONET (reference)", y = "Model", title = expression(paste("(c) Angstrom ", alpha))) +
  theme.common + theme(legend.position = "none", panel.spacing.x = unit(0.01, "npc"),
                        plot.margin = margin(2, 2, 2, 2, "pt"))

#################################################################################
# Output: single-page PDF via multiplot + layout_matrix
#################################################################################
dir.create(dirname(fig.path), recursive = TRUE, showWarnings = FALSE)

fig.h.total.mm <- fig.h.density.mm + 2 * fig.h.scatter.mm
p <- p.density / p.aod / p.alpha +
  plot_layout(heights = c(fig.h.density.mm, fig.h.scatter.mm, fig.h.scatter.mm))

ggsave(filename = fig.path, plot = p, width = fig.w.mm, height = fig.h.total.mm, units = "mm", dpi = 300)
cat(sprintf("Wrote: %s\n", fig.path))
