#!/usr/bin/env perl
use strict;
use warnings;

my %reward_by_step;
while (my $line = <>) {
    next unless $line =~ /training\/global_step:(\d+)/;
    my $step = $1;
    next unless $line =~ /critic\/score\/mean:([-+0-9.eE]+)/;
    $reward_by_step{$step} = 0 + $1;
}

print "range,count,mean_reward,implied_accuracy\n";
for my $start (1, 51, 101, 151, 201, 251) {
    my $end = $start + 49;
    my @values = map { $reward_by_step{$_} } grep { exists $reward_by_step{$_} } ($start .. $end);
    next unless @values;
    my $mean = 0;
    $mean += $_ for @values;
    $mean /= scalar @values;
    my $accuracy = ($mean + 1) / 2;
    print "$start-$end," . scalar(@values) . ",$mean,$accuracy\n";
}
