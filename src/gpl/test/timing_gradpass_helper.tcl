# Helper for timing gradient pass parameters (from commit 883c9767)
# Source this file in test scripts to use timing_gradpass parameters in global_placement
#
# Usage:
#   source timing_gradpass_helper.tcl
#   global_placement_with_timing_gradpass
#   OR
#   eval [concat global_placement $::gpl_base_args [get_timing_gradpass_args]]

# Check if the global timing gradient pass variables exist
# (they are set by regression-large script)
if { ![info exists ::gpl_timing_gradpass_top_n] } {
  set ::gpl_timing_gradpass_top_n 10
}
if { ![info exists ::gpl_timing_gradpass_proj_weight] } {
  set ::gpl_timing_gradpass_proj_weight 1.0
}
if { ![info exists ::gpl_timing_gradpass_end_to_end_weight] } {
  set ::gpl_timing_gradpass_end_to_end_weight 1.0
}
if { ![info exists ::gpl_timing_gradpass_slack_sharpness] } {
  set ::gpl_timing_gradpass_slack_sharpness 1.0
}
if { ![info exists ::gpl_timing_gradpass_slack_offset] } {
  set ::gpl_timing_gradpass_slack_offset 0.0
}

# Get timing gradient pass arguments as a list
proc get_timing_gradpass_args { } {
  return [list \
    -timing_gradpass_top_n $::gpl_timing_gradpass_top_n \
    -timing_gradpass_proj_weight $::gpl_timing_gradpass_proj_weight \
    -timing_gradpass_end_to_end_weight $::gpl_timing_gradpass_end_to_end_weight \
    -timing_gradpass_slack_sharpness $::gpl_timing_gradpass_slack_sharpness \
    -timing_gradpass_slack_offset $::gpl_timing_gradpass_slack_offset]
}

# Wrapper for global_placement that includes timing gradient pass arguments
proc global_placement_with_timing_gradpass { args } {
  eval global_placement $args [get_timing_gradpass_args]
}
