using Synchrony, Base.Test

# Test pxcorr
x = [rand() > 0.5 for i = 1:50]
y = [rand() > 0.5 for i = 1:50]
@test_approx_eq xcorr(x, y) pxcorr(find(x), find(y), -49:49)
