# fold_timer.R — Run times of the benchmark methods, written to fold_times.csv (columns fold, time_s),
# as fold_timer.py does for the Python scripts:
#
#   START <- Sys.time()                               # right after the library() calls (not timed)
#   fold_times <- c()
#   fold_times[as.character(i)] <- seconds of fold i  # e.g. as.numeric(difftime(end, start, units = "secs"))
#   write_fold_times(fold_times, START, out_dir)
#
# Rows: 'shared' = everything outside the folds (data loading, writing outputs), one row per fold
# (numbered like the script's prediction files), 'total' = shared + the folds.
write_fold_times <- function(fold_times, start, out_dir) {
  total <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(data.frame(fold = c("shared", names(fold_times), "total"),
                       time_s = sprintf("%.2f", c(total - sum(fold_times), fold_times, total))),
            file.path(out_dir, "fold_times.csv"), row.names = FALSE, quote = FALSE)
}
