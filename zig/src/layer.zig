const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const Array = @import("./array.zig").Array;
const NormPrng = @import("./rand.zig").NormPrng;

fn invsqrt(x: f64) f64 {
    var y: f64 = @bitCast(0x5fe6eb50c7b537a9 - (@as(u64, @bitCast(x)) >> 1));
    inline for (0..3) |_| y *= 1.5 - (0.5 * x * y * y);
    return y;
}

const LayerArgs = struct {
    sz_i: usize,
    sz_o: usize,
    prng: NormPrng,
};

const Dense = struct {
    w: [][]f64, // (nrow, ncol) = (n = 1:N, m = 1:M)
    x: []f64, // m = 1:M
    b: []f64, // n = 1:N

    fn init(Arr: *const Array, args: *const LayerArgs, alpha: f64) !Dense {
        const w: [][]f64 = try Arr.matrix(args.sz_o, args.sz_i);
        const x: []f64 = try Arr.vector(args.sz_i);
        const b: []f64 = try Arr.vector(args.sz_o);

        for (w) |row| {
            for (row) |*w_nm| {
                w_nm.* = alpha * args.prng.rand();
            }
        }

        @memset(b, 0.0);

        return .{ .w = w, .x = x, .b = b };
    }

    fn deinit(self: *const Dense, arr: Array) void {
        arr.free(self.w);
        arr.free(self.x);
        arr.free(self.b);
        return;
    }

    // z = W⋅x + b
    fn forward(self: *const Dense, z: []f64) void {
        // z ← b
        for (z, self.b) |*z_n, b_n| z_n.* = b_n;

        // z += W⋅x
        var tmp: f64 = undefined;
        for (z, self.w) |*z_n, row| {
            tmp = 0.0;
            for (self.x, row) |x_m, w_nm| {
                tmp += w_nm * x_m;
            }
            z_n.* += tmp;
        }

        return;
    }

    // dc/dx = Wᵀ ⋅ dc/dz
    fn backward(self: *const Dense, dcdx: []f64, dcdz: []f64) void {
        @memset(dcdx, 0.0);

        for (self.w, dcdz) |row, dcdz_n| {
            for (row, dcdx) |w_mn, *dcdx_n| {
                dcdx_n.* += w_mn * dcdz_n;
            }
        }

        return;
    }

    // b -= α ⋅ dc/db => b -= α ⋅ dc/dz
    // W -= α ⋅ dc/dW => W -= α ⋅ dc/dz ⋅ xᵀ
    fn learn(self: *const Dense, dcdz: []f64, learn_rate: f64) void {
        // b -= α ⋅ dc/dz
        for (self.b, dcdz) |*b_n, dcdz_n| {
            b_n.* -= learn_rate * dcdz_n;
        }
        // W -= α ⋅ dc/dz ⋅ xᵀ
        for (self.w, dcdz) |row, dcdz_n| {
            for (row, self.x) |*w_nm, x_m| {
                w_nm.* -= learn_rate * dcdz_n * x_m;
            }
        }
        return;
    }
};

test "Dense.init" {
    const ArrF64 = Array{ .allocator = testing.allocator };
    const prng: NormPrng = try NormPrng.init(testing.allocator, null);
    defer prng.deinit(testing.allocator);

    const args: LayerArgs = .{ .sz_i = 512, .sz_o = 512, .prng = prng };
    const alpha: f64 = invsqrt(0.5 * @as(f64, @floatFromInt(args.sz_i)));
    const dense: Dense = try Dense.init(&ArrF64, &args, alpha);
    defer dense.deinit(ArrF64);
}

test "Dense.forward, Dense.backward, and Dense.learn" {
    const ArrF64 = Array{ .allocator = testing.allocator };

    const W: [][]f64 = try ArrF64.matrix(3, 2);
    inline for (.{ 5.0, -2.0 }, W[0]) |v, *p| p.* = v;
    inline for (.{ 3.0, -5.0 }, W[1]) |v, *p| p.* = v;
    inline for (.{ 2.0, -3.0 }, W[2]) |v, *p| p.* = v;

    const b: []f64 = try ArrF64.vector(3);
    inline for (.{ -2.0, -5.0, -1.0 }, b) |v, *p| p.* = v;

    const x: []f64 = try ArrF64.vector(2);
    inline for (.{ 0.3, 0.5 }, x) |v, *p| p.* = v;

    const dense: Dense = .{ .w = W, .x = x, .b = b };
    defer dense.deinit(ArrF64);

    // test "Dense.forward"
    const z: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(z);

    dense.forward(z);
    try testing.expect(std.mem.eql(f64, &.{ -1.5, -6.6, -1.9 }, z));

    // test "Dense.backward"
    const dcdz: []f64 = try ArrF64.vector(3);
    inline for (.{ 0.2, 0.5, 0.3 }, dcdz) |v, *p| p.* = v;
    defer ArrF64.free(dcdz);

    const dcdx: []f64 = try ArrF64.vector(2);
    defer ArrF64.free(dcdx);

    dense.backward(dcdx, dcdz);
    try testing.expect(std.mem.eql(f64, &.{ 3.1, -3.8 }, dcdx));

    // test "Dense.learn"
    dense.learn(dcdz, 0.1);
    try testing.expect(std.mem.eql(f64, &.{ -2.02, -5.05, -1.03 }, dense.b));
    try testing.expect(std.mem.eql(f64, &.{ 4.994, -2.010 }, dense.w[0]));
    try testing.expect(std.mem.eql(f64, &.{ 2.985, -5.025 }, dense.w[1]));
    try testing.expect(std.mem.eql(f64, &.{ 1.991, -3.015 }, dense.w[2]));
}

const PReLU = struct {
    z: []f64, // n = 1:N
    a: []f64, // n = 1:N

    fn init(Arr: *const Array, args: *const LayerArgs, alpha: f64) !PReLU {
        const z: []f64 = try Arr.vector(args.sz_o);
        const a: []f64 = try Arr.vector(args.sz_o);

        @memset(a, alpha);

        return .{ .z = z, .a = a };
    }

    fn deinit(self: *const PReLU, arr: Array) void {
        arr.free(self.z);
        arr.free(self.a);
        return;
    }

    // y = f(z; a)
    fn forward(self: *const PReLU, y: []f64) void {
        for (y, self.z, self.a) |*y_n, z_n, a_n| {
            y_n.* = @max(0.0, z_n) + a_n * @min(0.0, z_n);
        }

        return;
    }

    // dc/dz ← dc/dy
    fn backward(self: *const PReLU, dcdz: []f64, dcdy: []f64) void {
        for (dcdz, dcdy, self.z, self.a) |*dcdz_n, dcdy_n, z_n, a_n| {
            dcdz_n.* = if (z_n > 0.0) dcdy_n else dcdy_n * a_n;
        }
        return;
    }

    // a -= α ⋅ dc/da
    fn learn(self: *const PReLU, dcdy: []f64, learn_rate: f64) void {
        for (self.a, self.z, dcdy) |*a_n, z_n, dcdy_n| {
            if (z_n < 0.0) a_n.* -= learn_rate * dcdy_n * z_n;
        }
        return;
    }
};

test "PReLU.forward, PReLU.backward, and PReLU.learn" {
    const ArrF64 = Array{ .allocator = testing.allocator };

    const z: []f64 = try ArrF64.vector(3);
    inline for (.{
        -0x1.7f2e48e8a71dfp-3,
        -0x1.4f0d844d013aap-3,
        -0x1.16872b020c49cp-5,
    }, z) |v, *p| p.* = v;

    const a: []f64 = try ArrF64.vector(3);
    inline for (.{
        0x1.851eb851eb852p-3,
        0x1.28f5c28f5c28fp-2,
        0x1.5c28f5c28f5c3p-3,
    }, a) |v, *p| p.* = v;

    const prelu: PReLU = .{ .z = z, .a = a };
    defer prelu.deinit(ArrF64);

    // test "PReLU.forward"
    const y: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(y);

    prelu.forward(y);
    try testing.expect(std.mem.eql(f64, y, &.{
        -0x1.2337a80cf9e39p-5,
        -0x1.84a9478c868bap-5,
        -0x1.7acc4ef88b978p-8,
    }));

    // test "PReLU.backward"
    const dcdy: []f64 = try ArrF64.vector(3);
    inline for (.{
        0x1.c28f5c28f5c29p-4,
        0x1.5c28f5c28f5c3p-3,
        0x1.0a3d70a3d70a4p-3,
    }, dcdy) |v, *p| p.* = v;
    defer ArrF64.free(dcdy);

    const dcdz: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(dcdz);

    prelu.backward(dcdz, dcdy);
    try testing.expect(std.mem.eql(f64, dcdz, &.{
        0x1.566cf41f212d8p-6,
        0x1.93dd97f62b6afp-5,
        0x1.6a161e4f765fep-6,
    }));

    // test "PReLU.learn"
    prelu.learn(dcdy, 0.05);
    try testing.expect(std.mem.eql(f64, prelu.a, &.{
        0x1.873a3d12b005fp-3,
        0x1.2a624c2572805p-2,
        0x1.5c9cd3e0bd44ap-3,
    }));
}
