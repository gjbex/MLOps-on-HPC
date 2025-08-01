library(yaml)

# The YAML file to read is specified as a command line argument
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("No YAML file specified. Please provide a file path as a command line argument.")
}
file_name <- args[1]

# Read the YAML file
config <- yaml.load_file(file_name)

# Use a value from the YAML configuration
print(config$split_data$random_state)
