#!/usr/bin/env Rscript

library(optparse)
library(reticulate)

option_list <- list(
  make_option(c("--a"), type = "double", default = 1.0, help = "Parameter a"),
  make_option(c("--t_max"), type = "integer", default = 10, help = "Maximum number of time steps")
)
parser <- OptionParser(option_list = option_list, description = "Run an experiment with specified parameters.")
opts <- parse_args(parser)

py_module_available("dvclive") || stop("Python module 'dvclive' is not available")
dvclive <- import("dvclive")

live <- dvclive$Live(report = "html")

compute <- function(t, a) {
  a * t^2
}

live$log_param("a", opts$a)
live$log_param("t_max", opts$t_max)

for (t in seq(0, opts$t_max)) {
  result <- compute(t, opts$a)
  live$log_metric("result", result)
  live$next_step()
}

live$end()