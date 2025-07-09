config param N = 6;

for i in 1..(N - 1) by -1 {
    writeln(i);
}

writeln((N - 1)..1 by -1);