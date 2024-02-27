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

    fn deinit(self: *const Dense, arr: Array) void {
        arr.free(self.w);
        arr.free(self.x);
        arr.free(self.b);
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
