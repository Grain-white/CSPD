#!/usr/bin/env perl
use strict;
use warnings;

my @datasets = ('math-500', 'aime-2024', 'aime-2025', 'hmmt-2025');
print "step," . join(',', @datasets) . "\n";
while (my $line = <>) {
    next unless $line =~ /step:(\d+)\s+-/;
    my $step = $1;
    my @values;
    my $complete = 1;
    for my $dataset (@datasets) {
        if ($line =~ /val-core\/\Q$dataset\E\/acc\/mean\@12:([-+0-9.eE]+)/) {
            push @values, $1;
        } else {
            $complete = 0;
            last;
        }
    }
    print join(',', $step, @values) . "\n" if $complete;
}
