#!/usr/bin/env perl
use strict;
use warnings;

while (<>) {
    # Match float fields in three formats:
    # 1. Normal decimal: :f:0.993724 or :f:0.99
    # 2. Leading dot: :f:.0549 (equivalent to 0.0549)
    # 3. Integer: :f:0 or :f:1 (needs to become 0.000 or 1.000)
    
    # Handle normal decimals: :f:0.993724 → :f:0.993
    s/(:f:)(\d+)\.(\d+)/$1 . $2 . "." . substr($3 . "000", 0, 3)/ge;
    
    # Handle leading dots: :f:.0549 → :f:0.054
    s/(:f:)\.(\d+)/$1 . "0." . substr($2 . "000", 0, 3)/ge;
    
    # Handle integers: :f:0 → :f:0.000, :f:1 → :f:1.000
    s/(:f:)(\d+)(\s|\t|$)/$1 . $2 . ".000" . $3/ge;
    
    print;
}
