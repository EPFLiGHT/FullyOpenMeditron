# Shared by the launchers (source it, do not run it).
#
# Optional Slurm reservation. Set RESERVATION to a reservation name to run the
# jobs in it; it is added only when it exists, since sbatch refuses a job that
# names a missing reservation. Unset: jobs go to the plain partition.

# usage: reservation_opts <account> <partition>
# Prints "--reservation=NAME" when it applies, nothing otherwise.
reservation_opts() {
    [ -n "${RESERVATION:-}" ] || return 0
    scontrol show reservation "$RESERVATION" >/dev/null 2>&1 || return 0
    printf -- '--reservation=%s\n' "$RESERVATION"
}
