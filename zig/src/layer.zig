const std = @import("std");
const mem = std.mem;

const Array = @import("./array.zig").Array;
const NormPrng = @import("./rand.zig").NormPrng;

const LayerArgs = struct {
    sz_i: usize,
    sz_o: usize,
    prng: *const NormPrng,
};

const Layer = struct {
    W: [][]f64, // weights
    b: []f64, // bias
    a: []f64, // PReLU params

    fn init(allocator: mem.Allocator, args: LayerArgs) !Layer {
        const ArrF64: Array = .{ .allocator = allocator };

        const W: [][]f64 = blk: {
            const mat: [][]f64 = try ArrF64.matrix(args.sz_o, args.sz_i);
            for (mat) |row| {
                for (row) |*ptr| ptr.* = args.prng.rand();
            }
            break :blk mat;
        };

        const b: []f64 = try ArrF64.vector(args.sz_o);
        const a: []f64 = try ArrF64.vector(args.sz_o);

        return .{ .W = W, .b = b, .a = a };
    }

    fn deinit(self: Layer, allocator: mem.Allocator) void {
        const ArrF64: Array = .{ .allocator = allocator };

        ArrF64.free(self.W);
        ArrF64.free(self.b);
        ArrF64.free(self.a);
        return;
    }

    // y = W⋅x + b
    fn forward() void {}
    fn backward() void {}
};

test "Layer.init & deinit" {
    std.debug.print("\n", .{});

    const prng: NormPrng = try NormPrng.init(std.testing.allocator, null);
    defer prng.free(std.testing.allocator);

    const layer: Layer = try Layer.init(
        std.testing.allocator,
        .{ .sz_i = 4, .sz_o = 5, .prng = &prng },
    );
    defer layer.deinit(std.testing.allocator);

    std.debug.print("[[ layer.W ]]\n", .{});
    for (layer.W) |row| std.debug.print("{any}\n", .{row});

    std.debug.print("[[ layer.b ]]\n{any}\n", .{layer.b});
    std.debug.print("[[ layer.a ]]\n{any}\n", .{layer.a});
}
