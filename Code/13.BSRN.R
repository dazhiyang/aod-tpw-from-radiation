#################################################################################
# 13.BSRN.R — global map of BSRN station locations (status: active / inactive / closed).
#
# Styling: Wong palette (SKILL.md), Times New Roman, 160 mm width, line width 0.3.
# Based on legacy ``map.r`` (ggrepel labels); coastlines via ``maps::map(interior = FALSE)``.
#
# Output default: ``Revision 1/BSRN_stations_map.pdf`` (override ``OUTPUT_FIG``).
#
# Requires: ggplot2, dplyr, maps (via borders), ggrepel
#################################################################################

rm(list = ls(all = TRUE))
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(maps)
  library(ggrepel)
})

.get.script.dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args)
  if (length(m) > 0) return(dirname(normalizePath(sub("^--file=", "", args[m[1]]))))
  if (!is.null(sys.frame(1)$ofile)) return(dirname(normalizePath(sys.frame(1)$ofile)))
  return(getwd())
}
dir0 <- normalizePath(file.path(.get.script.dir(), ".."), mustWork = FALSE)

plot.size <- 7
line.size <- 0.3
point.size <- 1.3
label.size <- plot.size*5/14
fig.w.mm <- as.numeric(Sys.getenv("FIG_W_MM", "160"))
fig.h.mm <- as.numeric(Sys.getenv("FIG_H_MM", "100"))
export.dpi <- as.numeric(Sys.getenv("EXPORT_DPI", "300"))

out.fig <- Sys.getenv(
  "OUTPUT_FIG",
  file.path(dir0, "Revision 1", "BSRN_stations_map.pdf")
)

# Wong discrete colours: active, inactive, closed (SKILL.md order 1–3).
col.active <- "#E69F00"
col.inactive <- "#56B4E9"
col.closed <- "#CC79A7"
status.colors <- c("active" = col.active, "inactive" = col.inactive, "closed" = col.closed)
status.shapes <- c("active" = 16L, "inactive" = 17L, "closed" = 15L)

.bsrn_row <- function(abbr, name, lat, lon, elev, status, kgc) {
  data.frame(
    abbr = abbr, name = name, lat = lat, lon = lon,
    elev = elev, status = status, kgc = kgc,
    stringsAsFactors = FALSE
  )
}

load_bsrn_stations <- function() {
  do.call(
    rbind,
    list(
      .bsrn_row("ABS", "Abashiri", 44.0178, 144.2797, 38.0, "active", "Dfb"),
      .bsrn_row("ALE", "Alert", 82.49, -62.42, 127.0, "closed", "ET"),
      .bsrn_row("ASP", "Alice Springs", -23.798, 133.888, 547.0, "inactive", "BWh"),
      .bsrn_row("BAR", "Barrow", 71.323, -156.607, 8.0, "active", "ET"),
      .bsrn_row("BER", "Bermuda", 32.3008, -64.766, 8.0, "active", "Af"),
      .bsrn_row("BIL", "Billings", 36.605, -97.516, 317.0, "active", "Cfa"),
      .bsrn_row("BON", "Bondville", 40.0667, -88.3667, 213.0, "active", "Dfa"),
      .bsrn_row("BOS", "Boulder (BOS)", 40.125, -105.237, 1689.0, "active", "BSk"),
      .bsrn_row("BOU", "Boulder (BOU)", 40.05, -105.007, 1577.0, "closed", "BSk"),
      .bsrn_row("BRB", "Brasilia", -15.601, -47.713, 1023.0, "inactive", "Aw"),
      .bsrn_row("BUD", "Budapest-Lorinc", 47.4291, 19.1822, 139.1, "active", "Dfb"),
      .bsrn_row("CAB", "Cabauw", 51.968, 4.928, 0.0, "active", "Cfb"),
      .bsrn_row("CAM", "Camborne", 50.2167, -5.3167, 88.0, "inactive", "Cfb"),
      .bsrn_row("CAP", "Cape Baranova", 79.27, 101.75, 25.0, "closed", "ET"),
      .bsrn_row("CAR", "Carpentras", 44.083, 5.059, 100.0, "closed", "Csa"),
      .bsrn_row("CLH", "Chesapeake Light", 36.905, -75.713, 37.0, "closed", "Cfa"),
      .bsrn_row("CNR", "Cener", 42.816, -1.601, 471.0, "active", "Cfb"),
      .bsrn_row("COC", "Cocos Island", -12.193, 96.835, 6.0, "inactive", "Af"),
      .bsrn_row("DAA", "De Aar", -30.6667, 23.993, 1287.0, "inactive", "BSk"),
      .bsrn_row("DAR", "Darwin", -12.425, 130.891, 30.0, "closed", "Aw"),
      .bsrn_row("DOM", "Concordia Station, Dome C", -75.1, 123.383, 3233.0, "active", "EF"),
      .bsrn_row("DRA", "Desert Rock", 36.626, -116.018, 1007.0, "active", "BWk"),
      .bsrn_row("DWN", "Darwin Met Office", -12.424, 130.8925, 32.0, "inactive", "Aw"),
      .bsrn_row("E13", "Southern Great Plains", 36.605, -97.485, 318.0, "active", "Cfa"),
      .bsrn_row("ENA", "Eastern North Atlantic", 39.0911, -28.0292, 15.2, "inactive", "Csa"),
      .bsrn_row("EUR", "Eureka", 79.989, -85.9404, 85.0, "closed", "ET"),
      .bsrn_row("FLO", "Florianopolis", -27.6047, -48.5227, 11.0, "active", "Cfa"),
      .bsrn_row("FPE", "Fort Peck", 48.3167, -105.1, 634.0, "active", "BSk"),
      .bsrn_row("FUA", "Fukuoka", 33.5822, 130.3764, 3.0, "closed", "Cfa"),
      .bsrn_row("GAN", "Gandhinagar", 23.1101, 72.6276, 65.0, "closed", "BSh"),
      .bsrn_row("GCR", "Goodwin Creek", 34.2547, -89.8729, 98.0, "active", "Cfa"),
      .bsrn_row("GIM", "Granite Island", 46.721, -87.411, 208.0, "active", "Dfb"),
      .bsrn_row("GOB", "Gobabeb", -23.5614, 15.042, 407.0, "active", "BWh"),
      .bsrn_row("GUR", "Gurgaon", 28.4249, 77.156, 259.0, "closed", "BSh"),
      .bsrn_row("GVN", "Georg von Neumayer", -70.65, -8.25, 42.0, "active", "EF"),
      .bsrn_row("HOW", "Howrah", 22.5535, 88.3064, 51.0, "closed", "Aw"),
      .bsrn_row("ILO", "Ilorin", 8.5333, 4.5667, 350.0, "closed", "Aw"),
      .bsrn_row("INO", "Marguele", 44.3439, 26.0123, 110.0, "active", "Dfa"),
      .bsrn_row("ISH", "Ishigakijima", 24.3367, 124.1644, 5.7, "active", "Af"),
      .bsrn_row("IZA", "Iza\u00f1a", 28.3093, -16.4993, 2372.9, "active", "Csb"),
      .bsrn_row("KWA", "Kwajalein", 8.72, 167.731, 10.0, "closed", "Af"),
      .bsrn_row("LAU", "Lauder", -45.045, 169.689, 350.0, "inactive", "Cfb"),
      .bsrn_row("LER", "Lerwick", 60.1389, -1.1847, 80.0, "inactive", "Cfc"),
      .bsrn_row("LIN", "Lindenberg", 52.21, 14.122, 125.0, "active", "Dfb"),
      .bsrn_row("LMP", "Lampedusa", 35.518, 12.63, 50.0, "active", "BSh"),
      .bsrn_row("LRC", "Langley Research Center", 37.1038, -76.3872, 3.0, "active", "Cfa"),
      .bsrn_row("LYU", "Lanyu Station", 22.037, 121.5583, 324.0, "active", "Af"),
      .bsrn_row("MAN", "Momote", -2.058, 147.425, 6.0, "closed", "Af"),
      .bsrn_row("MNM", "Minamitorishima", 24.2883, 153.9833, 7.1, "active", "Aw"),
      .bsrn_row("NAU", "Nauru Island", -0.521, 166.9167, 7.0, "closed", "Af"),
      .bsrn_row("NEW", "Newcastle", -32.8842, 151.7289, 18.5, "inactive", "Cfa"),
      .bsrn_row("NYA", "Ny-\u00c5lesund", 78.9227, 11.9273, 11.0, "active", "ET"),
      .bsrn_row("OHY", "Observatory of Huancayo", -12.05, -75.32, 3314.0, "active", "Cwb"),
      .bsrn_row("PAL", "Palaiseau, SIRTA Observatory", 48.713, 2.208, 156.0, "active", "Cfb"),
      .bsrn_row("PAR", "Paramaribo", 5.806, -55.2146, 4.0, "active", "Af"),
      .bsrn_row("PAY", "Payerne", 46.8123, 6.9422, 491.0, "active", "Dfb"),
      .bsrn_row("PSU", "Rock Springs", 40.72, -77.9333, 376.0, "active", "Dfa"),
      .bsrn_row("PTR", "Petrolina", -9.069, -40.32, 387.0, "inactive", "BSh"),
      .bsrn_row("QIQ", "Qiqihar", 47.7957, 124.4852, 170.0, "active", "Dwa"),
      .bsrn_row("REG", "Regina", 50.205, -104.713, 578.0, "closed", "Dfb"),
      .bsrn_row("RLM", "Rolim de Moura", -11.582, -61.773, 252.0, "closed", "Aw"),
      .bsrn_row("RUN", "Reunion Island, University", -20.9014, 55.4836, 116.0, "active", "Am"),
      .bsrn_row("SAP", "Sapporo", 43.06, 141.3286, 17.2, "closed", "Dfb"),
      .bsrn_row("SBO", "Sede Boqer", 30.8597, 34.7794, 500.0, "closed", "BWh"),
      .bsrn_row("SEL", "Selegua", 15.784, -91.9902, 602.0, "active", "Aw"),
      .bsrn_row("SMS", "S\u00e3o Martinho da Serra", -29.4428, -53.8231, 489.0, "inactive", "Cfa"),
      .bsrn_row("SON", "Sonnblick", 47.054, 12.9577, 3108.9, "active", "ET"),
      .bsrn_row("SOV", "Solar Village", 24.91, 46.41, 650.0, "closed", "BWh"),
      .bsrn_row("SPO", "South Pole", -89.983, -24.799, 2800.0, "active", "EF"),
      .bsrn_row("SXF", "Sioux Falls", 43.73, -96.62, 473.0, "active", "Dfa"),
      .bsrn_row("SYO", "Syowa", -69.0053, 39.5811, 29.0, "active", "EF"),
      .bsrn_row("TAM", "Tamanrasset", 22.7903, 5.5292, 1385.0, "active", "BWh"),
      .bsrn_row("TAT", "Tateno", 36.0581, 140.1258, 25.0, "active", "Cfa"),
      .bsrn_row("TIK", "Tiksi", 71.5862, 128.9188, 48.0, "closed", "ET"),
      .bsrn_row("TIR", "Tiruvallur", 13.0923, 79.9738, 36.0, "closed", "Aw"),
      .bsrn_row("TOR", "Toravere", 58.2641, 26.4613, 70.0, "active", "Dfb"),
      .bsrn_row("XIA", "Xianghe", 39.754, 116.962, 32.0, "closed", "Dwa"),
      .bsrn_row("YUS", "Yushan Station", 23.4876, 120.9595, 3858.0, "active", "ET")
    )
  )
}

bsrn <- load_bsrn_stations()
bsrn$status <- factor(bsrn$status, levels = c("active", "inactive", "closed"))

# Coastlines only (no country fill; no internal borders): maps::map(interior = FALSE).
coastline_df <- function() {
  w <- maps::map("world", fill = FALSE, interior = FALSE, plot = FALSE)
  df <- data.frame(long = w$x, lat = w$y)
  df$group <- cumsum(is.na(df$long) | is.na(df$lat))
  df[stats::complete.cases(df), , drop = FALSE]
}

coast <- coastline_df()

p.map <- ggplot() +
  geom_path(
    data = coast,
    aes(x = long, y = lat, group = group),
    colour = "grey55",
    linewidth = line.size
  ) +
  geom_point(
    data = bsrn,
    aes(x = lon, y = lat, colour = status, shape = status),
    size = point.size,
    stroke = line.size
  ) +
  ggrepel::geom_text_repel(
    data = bsrn,
    aes(x = lon, y = lat, label = abbr),
    size = label.size,
    colour = "black",
    family = "serif",
    segment.size = line.size,
    segment.colour = "grey40",
    max.overlaps = Inf,
    min.segment.length = 0,
    box.padding = 0.15,
    point.padding = 0.2
  ) +
  coord_fixed(ratio = 1.3, xlim = c(-180, 180), ylim = c(-90, 90), expand = FALSE) +
  scale_colour_manual(
    values = status.colors,
    labels = c("Active", "Inactive", "Closed"),
    name = "Status"
  ) +
  scale_shape_manual(
    values = status.shapes,
    labels = c("Active", "Inactive", "Closed"),
    name = "Status"
  ) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = plot.size, base_family = "serif") +
  theme(
    text = element_text(family = "serif", size = plot.size),
    plot.title = element_text(family = "serif", size = plot.size),
    legend.text = element_text(family = "serif", size = plot.size),
    legend.title = element_text(family = "serif", size = plot.size),
    panel.grid = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    legend.position = c(0.12, 0.22),
    legend.background = element_blank(),
    legend.key.size = unit(plot.size * 1.1, "pt"),
    plot.margin = margin(2, 2, 2, 2, "pt"),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA)
  )

dir.create(dirname(out.fig), recursive = TRUE, showWarnings = FALSE)
ggsave(out.fig, plot = p.map, width = fig.w.mm, height = fig.h.mm, units = "mm", dpi = export.dpi)
cat(sprintf("Wrote: %s (%d stations)\n", out.fig, nrow(bsrn)))
