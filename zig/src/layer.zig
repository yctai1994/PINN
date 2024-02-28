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
};

pub const Layer = struct {
    dense: Dense,
    prelu: PReLU,

    pub fn init(
        allocator: std.mem.Allocator,
        comptime args: *const LayerArgs,
        prng: *const NormPrng,
    ) !Layer {
        const ArrF64 = Array{ .allocator = allocator };
        const alpha: f64 = invsqrt(0.5 * @as(f64, @floatFromInt(args.sz_i)));

        const dense = try Dense.init(&ArrF64, args, prng, alpha);
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

    pub fn forward(self: *const Layer, uvec: []f64) void {
        self.dense.forward(self.prelu.zvec);
        self.prelu.forward(uvec);
        return;
    }

    pub fn backward(self: *const Layer, dcdx: []f64, learn_rate: f64) void {
        self.prelu.backward(self.dense.dcdz);
        self.prelu.learn(learn_rate);
        self.dense.backward(dcdx);
        self.dense.learn(learn_rate);
        return;
    }
};

const Dense = struct {
    wmat: [][]f64, // (nrow, ncol) = (n = 1:N, m = 1:M)
    xvec: []f64, // m = 1:M
    bvec: []f64, // n = 1:N
    dcdz: []f64, // n = 1:N

    fn init(arr: *const Array, args: *const LayerArgs, prng: *const NormPrng, alpha: f64) !Dense {
        const wmat: [][]f64 = try arr.matrix(args.sz_o, args.sz_i);
        errdefer arr.free(wmat);

        const xvec: []f64 = try arr.vector(args.sz_i);
        errdefer arr.free(xvec);

        const bvec: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(bvec);

        const dcdz: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(dcdz);

        for (wmat) |row| {
            for (row) |*w_nm| {
                w_nm.* = alpha * prng.rand();
            }
        }

        @memset(bvec, 0.0);

        return .{ .wmat = wmat, .xvec = xvec, .bvec = bvec, .dcdz = dcdz };
    }

    fn deinit(self: *const Dense, arr: *const Array) void {
        arr.free(self.wmat);
        arr.free(self.xvec);
        arr.free(self.bvec);
        arr.free(self.dcdz);
        return;
    }

    // z = W⋅x + b
    fn forward(self: *const Dense, zvec: []f64) void {
        // z ← b
        for (zvec, self.bvec) |*z_n, b_n| {
            z_n.* = b_n;
        }

        // z += W⋅x
        var tmp: f64 = undefined;
        for (zvec, self.wmat) |*z_n, row| {
            tmp = 0.0;
            for (self.xvec, row) |x_m, w_nm| {
                tmp += w_nm * x_m;
            }
            z_n.* += tmp;
        }

        return;
    }

    // dc/dx = Wᵀ ⋅ dc/dz
    fn backward(self: *const Dense, dcdx: []f64) void {
        @memset(dcdx, 0.0);

        for (self.wmat, self.dcdz) |row, dcdz_n| {
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
        for (self.bvec, self.dcdz) |*b_n, dcdz_n| {
            b_n.* -= learn_rate * dcdz_n;
        }

        // W -= α ⋅ dc/dz ⋅ xᵀ
        for (self.wmat, self.dcdz) |row, dcdz_n| {
            for (row, self.xvec) |*w_nm, x_m| {
                w_nm.* -= learn_rate * dcdz_n * x_m;
            }
        }

        return;
    }
};

const PReLU = struct {
    zvec: []f64, // n = 1:N
    avec: []f64, // n = 1:N
    dcdu: []f64, // n = 1:N

    fn init(arr: *const Array, args: *const LayerArgs, alpha: f64) !PReLU {
        const zvec: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(zvec);

        const avec: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(avec);

        const dcdu: []f64 = try arr.vector(args.sz_o);
        errdefer arr.free(dcdu);

        @memset(avec, alpha);

        return .{ .zvec = zvec, .avec = avec, .dcdu = dcdu };
    }

    fn deinit(self: *const PReLU, arr: *const Array) void {
        arr.free(self.zvec);
        arr.free(self.avec);
        arr.free(self.dcdu);
        return;
    }

    // u = f(z; a)
    fn forward(self: *const PReLU, uvec: []f64) void {
        for (uvec, self.zvec, self.avec) |*u_n, z_n, a_n| {
            u_n.* = @max(0.0, z_n) + a_n * @min(0.0, z_n);
        }

        return;
    }

    // dc/dz ← dc/du
    fn backward(self: *const PReLU, dcdz: []f64) void {
        for (dcdz, self.dcdu, self.zvec, self.avec) |*dcdz_n, dcdu_n, z_n, a_n| {
            dcdz_n.* = if (z_n > 0.0) dcdu_n else dcdu_n * a_n;
        }
        return;
    }

    // a -= α ⋅ dc/da
    fn learn(self: *const PReLU, learn_rate: f64) void {
        for (self.avec, self.zvec, self.dcdu) |*a_n, z_n, dcdu_n| {
            if (z_n < 0.0) a_n.* -= learn_rate * dcdu_n * z_n;
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
        &.{ .sz_i = 2, .sz_o = 3 },
        &prng,
    );
    errdefer layer.deinit(std.testing.allocator);

    const uvec: []f64 = try ArrF64.vector(3);
    errdefer ArrF64.free(uvec);

    const dcdx: []f64 = try ArrF64.vector(2);
    errdefer ArrF64.free(dcdx);

    defer {
        prng.deinit(std.testing.allocator);
        layer.deinit(std.testing.allocator);
        ArrF64.free(uvec);
        ArrF64.free(dcdx);
    }

    inline for (layer.dense.wmat[0], .{
        0x1.47ae147ae147bp-6,
        -0x1.1eb851eb851ecp-4,
    }) |*p, v| p.* = v;

    inline for (layer.dense.wmat[1], .{
        0x1.28f5c28f5c28fp-2,
        -0x1.0a3d70a3d70a4p-3,
    }) |*p, v| p.* = v;

    inline for (layer.dense.wmat[2], .{
        0x1.c28f5c28f5c29p-4,
        -0x1.eb851eb851eb8p-6,
    }) |*p, v| p.* = v;

    inline for (layer.dense.bvec, .{
        -0x1.5c28f5c28f5c3p-3,
        -0x1.851eb851eb852p-3,
        -0x1.999999999999ap-5,
    }) |*p, v| p.* = v;

    inline for (layer.dense.xvec, .{
        0x1.d70a3d70a3d71p-3,
        0x1.3d70a3d70a3d7p-2,
    }) |*p, v| p.* = v;

    inline for (layer.prelu.avec, .{
        0x1.851eb851eb852p-3,
        0x1.28f5c28f5c28fp-2,
        0x1.5c28f5c28f5c3p-3,
    }) |*p, v| p.* = v;

    inline for (layer.prelu.dcdu, .{
        0x1.c28f5c28f5c29p-4,
        0x1.5c28f5c28f5c3p-3,
        0x1.0a3d70a3d70a4p-3,
    }) |*p, v| p.* = v;

    // test "Layer.forward"
    layer.forward(uvec);

    try std.testing.expect(std.mem.eql(f64, uvec, &.{
        -0x1.2337a80cf9e39p-5,
        -0x1.84a9478c868bap-5,
        -0x1.7acc4ef88b978p-8,
    }));

    // test "Layer.backward"
    layer.backward(dcdx, 0.05);

    try std.testing.expect(std.mem.eql(f64, layer.dense.wmat[0], &.{
        0x1.43bdfaa1f22dbp-6,
        -0x1.200c019602357p-4,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.dense.wmat[1], &.{
        0x1.2861233086b08p-2,
        -0x1.0bce131de9f5dp-3,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.dense.wmat[2], &.{
        0x1.c184dd49f2926p-4,
        -0x1.f121e0fb57fa9p-6,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.dense.bvec, &.{
        -0x1.5e4cd74927914p-3,
        -0x1.8a2b1704ff434p-3,
        -0x1.a2a6f3f52fc27p-5,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.dense.dcdz, &.{
        0x1.566cf41f212d8p-6,
        0x1.93dd97f62b6afp-5,
        0x1.6a161e4f765fep-6,
    }));

    try std.testing.expect(std.mem.eql(f64, layer.prelu.avec, &.{
        0x1.873a3d12b005fp-3,
        0x1.2a624c2572805p-2,
        0x1.5c9cd3e0bd44ap-3,
    }));

    try std.testing.expect(std.mem.eql(f64, dcdx, &.{
        0x1.18eb8950763a2p-6,
        -0x1.17acc4ef88b98p-7,
    }));
}
