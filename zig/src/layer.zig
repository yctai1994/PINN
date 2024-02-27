const std = @import("std");
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
    prng: *const NormPrng,
};

pub const Layer = struct {
    dense: Dense,
    prelu: PReLU,

    pub fn init(allocator: std.mem.Allocator, args: *const LayerArgs) !Layer {
        const ArrF64 = Array{ .allocator = allocator };
        const alpha: f64 = invsqrt(0.5 * @as(f64, @floatFromInt(args.sz_i)));

        const dense = try Dense.init(&ArrF64, args, alpha);
        errdefer dense.deinit(&ArrF64);

        const prelu = try PReLU.init(&ArrF64, args, alpha);
        errdefer prelu.deinit(&ArrF64);

        return .{ .dense = dense, .prelu = prelu };
    }

    pub fn deinit(self: *const Layer, allocator: std.mem.Allocator) void {
        const ArrF64 = Array{ .allocator = allocator };
        self.dense.deinit(&ArrF64);
        self.prelu.deinit(&ArrF64);
        return;
    }

    pub fn forward(self: *const Layer, y: []f64) void {
        self.dense.forward(self.prelu.z);
        self.prelu.forward(y);
        return;
    }

    pub fn backward(self: *const Layer, dcdx: []f64, learn_rate: f64) void {
        self.prelu.backward(self.dense.d);
        self.prelu.learn(learn_rate);
        self.dense.backward(dcdx);
        self.dense.learn(learn_rate);
        return;
    }
};

const Dense = struct {
    w: [][]f64, // (nrow, ncol) = (n = 1:N, m = 1:M)
    x: []f64, // m = 1:M
    b: []f64, // n = 1:N
    d: []f64, // dc/dz, n = 1:N

    fn init(arr: *const Array, args: *const LayerArgs, alpha: f64) !Dense {
        const w: [][]f64 = try arr.matrix(args.sz_o, args.sz_i);
        errdefer arr.free(w);

        const x: []f64 = try arr.vector(args.sz_i);
        errdefer arr.free(x);

        const b: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(b);

        const d: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(d);

        for (w) |row| {
            for (row) |*w_nm| {
                w_nm.* = alpha * args.prng.rand();
            }
        }

        @memset(b, 0.0);

        return .{ .w = w, .x = x, .b = b, .d = d };
    }

    fn deinit(self: *const Dense, arr: *const Array) void {
        arr.free(self.w);
        arr.free(self.x);
        arr.free(self.b);
        arr.free(self.d);
        return;
    }

    // z = W⋅x + b
    fn forward(self: *const Dense, z: []f64) void {
        // z ← b
        for (z, self.b) |*z_n, b_n| {
            z_n.* = b_n;
        }

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
    fn backward(self: *const Dense, dcdx: []f64) void {
        @memset(dcdx, 0.0);

        for (self.w, self.d) |row, dcdz_n| {
            for (row, dcdx) |w_mn, *dcdx_n| {
                dcdx_n.* += w_mn * dcdz_n;
            }
        }

        return;
    }

    // b -= α ⋅ dc/db => b -= α ⋅ dc/dz
    // W -= α ⋅ dc/dW => W -= α ⋅ dc/dz ⋅ xᵀ
    fn learn(self: *const Dense, learn_rate: f64) void {
        // b -= α ⋅ dc/dz
        for (self.b, self.d) |*b_n, dcdz_n| {
            b_n.* -= learn_rate * dcdz_n;
        }

        // W -= α ⋅ dc/dz ⋅ xᵀ
        for (self.w, self.d) |row, dcdz_n| {
            for (row, self.x) |*w_nm, x_m| {
                w_nm.* -= learn_rate * dcdz_n * x_m;
            }
        }

        return;
    }
};

const PReLU = struct {
    z: []f64, // n = 1:N
    a: []f64, // n = 1:N
    d: []f64, // dc/dy, n = 1:N

    fn init(arr: *const Array, args: *const LayerArgs, alpha: f64) !PReLU {
        const z: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(z);

        const a: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(a);

        const d: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(d);

        @memset(a, alpha);

        return .{ .z = z, .a = a, .d = d };
    }

    fn deinit(self: *const PReLU, arr: *const Array) void {
        arr.free(self.z);
        arr.free(self.a);
        arr.free(self.d);
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
    fn backward(self: *const PReLU, dcdz: []f64) void {
        for (dcdz, self.d, self.z, self.a) |*dcdz_n, dcdy_n, z_n, a_n| {
            dcdz_n.* = if (z_n > 0.0) dcdy_n else dcdy_n * a_n;
        }
        return;
    }

    // a -= α ⋅ dc/da
    fn learn(self: *const PReLU, learn_rate: f64) void {
        for (self.a, self.z, self.d) |*a_n, z_n, dcdy_n| {
            if (z_n < 0.0) a_n.* -= learn_rate * dcdy_n * z_n;
        }

        return;
    }
};

test "Layer.init, Layer.deinit, Layer.forward, Layer.backward, and Layer.learn" {
    const ArrF64 = Array{ .allocator = std.testing.allocator };
    const prng: NormPrng = try NormPrng.init(std.testing.allocator, null);
    errdefer prng.deinit(std.testing.allocator);

    const layer = try Layer.init(
        std.testing.allocator,
        &.{ .sz_i = 2, .sz_o = 3, .prng = &prng },
    );
    errdefer layer.deinit(std.testing.allocator);

    const y: []f64 = try ArrF64.vector(3);
    errdefer ArrF64.free(y);

    const dcdx: []f64 = try ArrF64.vector(2);
    errdefer ArrF64.free(dcdx);

    defer {
        prng.deinit(std.testing.allocator);
        layer.deinit(std.testing.allocator);
        ArrF64.free(y);
        ArrF64.free(dcdx);
    }

    inline for (layer.dense.w[0], .{
        0x1.47ae147ae147bp-6,
        -0x1.1eb851eb851ecp-4,
    }) |*p, v| p.* = v;

    inline for (layer.dense.w[1], .{
        0x1.28f5c28f5c28fp-2,
        -0x1.0a3d70a3d70a4p-3,
    }) |*p, v| p.* = v;

    inline for (layer.dense.w[2], .{
        0x1.c28f5c28f5c29p-4,
        -0x1.eb851eb851eb8p-6,
    }) |*p, v| p.* = v;

    inline for (layer.dense.b, .{
        -0x1.5c28f5c28f5c3p-3,
        -0x1.851eb851eb852p-3,
        -0x1.999999999999ap-5,
    }) |*p, v| p.* = v;

    inline for (layer.dense.x, .{
        0x1.d70a3d70a3d71p-3,
        0x1.3d70a3d70a3d7p-2,
    }) |*p, v| p.* = v;

    inline for (layer.prelu.a, .{
        0x1.851eb851eb852p-3,
        0x1.28f5c28f5c28fp-2,
        0x1.5c28f5c28f5c3p-3,
    }) |*p, v| p.* = v;

    inline for (layer.prelu.d, .{ // dc/dy
        0x1.c28f5c28f5c29p-4,
        0x1.5c28f5c28f5c3p-3,
        0x1.0a3d70a3d70a4p-3,
    }) |*p, v| p.* = v;

    // test "Layer.forward"
    layer.forward(y);

    try std.testing.expect(std.mem.eql(f64, y, &.{
        -0x1.2337a80cf9e39p-5,
        -0x1.84a9478c868bap-5,
        -0x1.7acc4ef88b978p-8,
    }));

    // test "Layer.backward"
    layer.backward(dcdx, 0.05);

    try std.testing.expect(std.mem.eql(f64, layer.dense.w[0], &.{
        0x1.43bdfaa1f22dbp-6,
        -0x1.200c019602357p-4,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.dense.w[1], &.{
        0x1.2861233086b08p-2,
        -0x1.0bce131de9f5dp-3,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.dense.w[2], &.{
        0x1.c184dd49f2926p-4,
        -0x1.f121e0fb57fa9p-6,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.dense.b, &.{
        -0x1.5e4cd74927914p-3,
        -0x1.8a2b1704ff434p-3,
        -0x1.a2a6f3f52fc27p-5,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.dense.d, &.{
        0x1.566cf41f212d8p-6,
        0x1.93dd97f62b6afp-5,
        0x1.6a161e4f765fep-6,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.prelu.a, &.{
        0x1.873a3d12b005fp-3,
        0x1.2a624c2572805p-2,
        0x1.5c9cd3e0bd44ap-3,
    }));

    try std.testing.expect(std.mem.eql(f64, dcdx, &.{
        0x1.18eb8950763a2p-6,
        -0x1.17acc4ef88b98p-7,
    }));
}
