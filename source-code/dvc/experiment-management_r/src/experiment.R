library(reticulate)

# Load the Python module
py_module_available("dvclive") || stop("Python module 'dvclive' is not available")
dvclive <- import("dvclive")

# Initialize dvclive
live <- dvclive$Live(report='html')

# delta is a command line parameter, initialize and log it
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) == 0) {
  stop("No command line arguments provided. Please provide a value for 'delta'.")
}
delta <- as.numeric(arguments[1])
live$log_param("delta", delta)

# Simulate a process and log metrics
current_score <- 0.0
live$log_metric("score", current_score)
live$next_step()
for(step in 1:100) {
  current_score <- current_score + delta
  live$log_metric("score", current_score)
  live$next_step()
}

live$end()  # finalize logs
