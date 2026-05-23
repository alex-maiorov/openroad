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
if { ![info exists ::gpl_timing_gradpass_slack_upper] } {
  set ::gpl_timing_gradpass_slack_upper 0.0
}
if { ![info exists ::gpl_timing_gradpass_sta_run_interval] } {
  set ::gpl_timing_gradpass_sta_run_interval 10
}
if { ![info exists ::gpl_timing_gradpass_saturation_kl] } {
  set ::gpl_timing_gradpass_saturation_kl 3.0
}
if { ![info exists ::gpl_timing_gradpass_saturation_minl] } {
  set ::gpl_timing_gradpass_saturation_minl 1000.0
}
if { ![info exists ::gpl_timing_gradpass_precond_count_weight] } {
  set ::gpl_timing_gradpass_precond_count_weight 1.0
}

# Get timing gradient pass arguments as a list
proc get_timing_gradpass_args { } {
  return [list \
    -timing_gradpass_top_n $::gpl_timing_gradpass_top_n \
    -timing_gradpass_proj_weight $::gpl_timing_gradpass_proj_weight \
    -timing_gradpass_end_to_end_weight $::gpl_timing_gradpass_end_to_end_weight \
     -timing_gradpass_slack_sharpness $::gpl_timing_gradpass_slack_sharpness \
     -timing_gradpass_slack_offset $::gpl_timing_gradpass_slack_offset \
     -timing_gradpass_slack_upper $::gpl_timing_gradpass_slack_upper \
     -timing_gradpass_sta_run_interval $::gpl_timing_gradpass_sta_run_interval \
     -timing_gradpass_saturation_kl $::gpl_timing_gradpass_saturation_kl \
     -timing_gradpass_saturation_minl $::gpl_timing_gradpass_saturation_minl \
     -timing_gradpass_precond_count_weight $::gpl_timing_gradpass_precond_count_weight]
}

# Wrapper for global_placement that includes timing gradient pass arguments
proc global_placement_with_timing_gradpass { args } {
  eval global_placement $args [get_timing_gradpass_args]
}
