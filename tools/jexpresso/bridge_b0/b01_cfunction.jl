# The closure-@cfunction check, as banked in JEXPRESSO_ENVIRONMENT.md section 4.2.
#
# Bridge roadmap B0.1, rung 2. A plain @cfunction works everywhere; a closure
# @cfunction needs a run-time trampoline, which aborted every adaptive
# Jexpresso run on this Mac in September 2026. This file is included by a
# plain Julia process and by the runtime juliacall starts inside Python.
# A healthy runtime prints 5 and then 7. Each line is flushed before the next
# call, so a log that ends after "plain cfunction" names the call that died.
#
# The plain target is a global function and the closure is built inside a
# function: other arrangements fail for unrelated reasons.

println(VERSION, " ", Sys.ARCH)
b01_g(x::Int) = x + 1
function b01_test_plain()
    cf = @cfunction(b01_g, Int, (Int,))
    return ccall(cf, Int, (Int,), 4)
end
function b01_test_closure(k)
    f = x -> x + k
    cf = @cfunction($f, Int, (Int,))
    return ccall(cf, Int, (Int,), 4)
end
println("plain cfunction: ", b01_test_plain())
flush(stdout)
println("closure cfunction: ", b01_test_closure(3))
flush(stdout)
