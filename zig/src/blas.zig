const std = @import("std");
const testing = std.testing;

const Matrix = @import("./array.zig").Matrix;
const Vector = @import("./array.zig").Vector;

const Transpose = enum(u1) { N, T }; // (N, T): (Normal, Transpose)
const WriteMode = enum(u1) { W, U }; // (W, U): (Overwrite, Update)

pub fn copy(des: []f64, src: []f64) void {
    for (des, src) |*pd, vs| pd.* = vs;
    return;
}

test "blas.copy" {
    const VecF64: Vector = .{ .allocator = testing.allocator };
    const src: []f64 = try VecF64.alloc(10);
    const des: []f64 = try VecF64.alloc(10);

    defer {
        VecF64.free(src);
        VecF64.free(des);
    }

    for (src, 0..) |*p, i| p.* = @floatFromInt(i);

    copy(des, src);

    for (des, src) |vd, vs| try testing.expectEqual(vd, vs);
}

pub fn xoty(x: []f64, y: []f64, z: []f64) void {
    for (x, y, z) |vx, vy, *pz| pz.* = vx * vy;
    return;
}

test "blas.xoty" {
    const VecF64: Vector = .{ .allocator = testing.allocator };
    const x: []f64 = try VecF64.alloc(10);
    const y: []f64 = try VecF64.alloc(10);
    const z: []f64 = try VecF64.alloc(10);

    defer {
        VecF64.free(x);
        VecF64.free(y);
        VecF64.free(z);
    }

    for (x, y, 0..) |*px, *py, i| {
        px.* = @floatFromInt(2 * i);
        py.* = @floatFromInt(3 * i);
    }

    xoty(x, y, z);

    for (z, 0..) |v, i| try testing.expectEqual(v, @as(f64, @floatFromInt(6 * i * i)));
}

pub fn gemv(comptime tA: Transpose, a: f64, A: [][]f64, x: []f64, b: f64, y: []f64, z: []f64) void {
    if (b == 0.0) {
        for (z) |*pz| pz.* = 0.0;
    } else if (b == 1.0) {
        for (y, z) |vy, *pz| pz.* = vy;
    } else if (b == -1.0) {
        for (y, z) |vy, *pz| pz.* = -vy;
    } else {
        for (y, z) |vy, *pz| pz.* = b * vy;
    }

    if (a == 0.0) return;

    switch (tA) {
        .N => {
            var tmp: f64 = undefined;
            if (a == 1.0) {
                for (A, z) |row, *pz| {
                    tmp = 0.0;
                    for (row, x) |vr, vx| tmp += vr * vx;
                    pz.* += tmp;
                }
            } else if (a == -1.0) {
                for (A, z) |row, *pz| {
                    tmp = 0.0;
                    for (row, x) |vr, vx| tmp -= vr * vx;
                    pz.* += tmp;
                }
            } else {
                for (A, z) |row, *pz| {
                    tmp = 0.0;
                    for (row, x) |vr, vx| tmp += a * vr * vx;
                    pz.* += tmp;
                }
            }
        },
        .T => {
            if (a == 1.0) {
                for (A, x) |row, vx| {
                    for (row, z) |vr, *pz| pz.* += vx * vr;
                }
            } else if (a == -1.0) {
                for (A, x) |row, vx| {
                    for (row, z) |vr, *pz| pz.* -= vx * vr;
                }
            } else {
                var tmp: f64 = undefined;
                for (A, x) |row, vx| {
                    tmp = a * vx;
                    for (row, z) |vr, *pz| pz.* += tmp * vr;
                }
            }
        },
    }

    return;
}

test "blas.gemv.N" {
    const MatF64: Matrix = .{ .allocator = testing.allocator };
    const A: [][]f64 = try MatF64.alloc(4, 3);
    inline for (.{ 4.0, 3.0, 1.0 }, A[0]) |v, *p| p.* = v;
    inline for (.{ 3.0, 7.0, 0.0 }, A[1]) |v, *p| p.* = v;
    inline for (.{ 2.0, 5.0, 3.0 }, A[2]) |v, *p| p.* = v;
    inline for (.{ 1.0, 1.0, 2.0 }, A[3]) |v, *p| p.* = v;

    const VecF64: Vector = .{ .allocator = testing.allocator };
    const x: []f64 = try VecF64.alloc(3);
    inline for (.{ 3.0, 1.0, 5.0 }, x) |v, *p| p.* = v;

    const y: []f64 = try VecF64.alloc(4);
    inline for (.{ -1.0, 3.0, -5.0, 1.0 }, y) |v, *p| p.* = v;

    const z: []f64 = try VecF64.alloc(4);

    defer {
        MatF64.free(A, 4, 3);
        VecF64.free(x);
        VecF64.free(y);
        VecF64.free(z);
    }

    const a_list: [4]f64 = .{ 0.0, 1.0, -1.0, 3.0 };
    const b_list: [4]f64 = .{ 0.0, 1.0, -1.0, -5.0 };
    const z_list: [16][4]f64 = .{
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ -1.0, 3.0, -5.0, 1.0 },
        .{ 1.0, -3.0, 5.0, -1.0 },
        .{ 5.0, -15.0, 25.0, -5.0 },
        .{ 20.0, 16.0, 26.0, 14.0 },
        .{ 19.0, 19.0, 21.0, 15.0 },
        .{ 21.0, 13.0, 31.0, 13.0 },
        .{ 25.0, 1.0, 51.0, 9.0 },
        .{ -20.0, -16.0, -26.0, -14.0 },
        .{ -21.0, -13.0, -31.0, -13.0 },
        .{ -19.0, -19.0, -21.0, -15.0 },
        .{ -15.0, -31.0, -1.0, -19.0 },
        .{ 60.0, 48.0, 78.0, 42.0 },
        .{ 59.0, 51.0, 73.0, 43.0 },
        .{ 61.0, 45.0, 83.0, 41.0 },
        .{ 65.0, 33.0, 103.0, 37.0 },
    };

    for (a_list, 0..) |a, ia| {
        for (b_list, 0..) |b, ib| {
            gemv(.N, a, A, x, b, y, z);

            try testing.expect(std.mem.eql(f64, &z_list[4 * ia + ib], z));
        }
    }
}

test "blas.gemv.T" {
    const MatF64: Matrix = .{ .allocator = testing.allocator };
    const A: [][]f64 = try MatF64.alloc(4, 3);
    inline for (.{ 7.0, -11.0, 0.0 }, A[0]) |v, *p| p.* = v;
    inline for (.{ 5.0, -17.0, 3.0 }, A[1]) |v, *p| p.* = v;
    inline for (.{ 1.0, -13.0, 2.0 }, A[2]) |v, *p| p.* = v;
    inline for (.{ 2.0, -19.0, 0.0 }, A[3]) |v, *p| p.* = v;

    const VecF64: Vector = .{ .allocator = testing.allocator };
    const x: []f64 = try VecF64.alloc(4);
    inline for (.{ 3.0, 1.0, 0.0, 5.0 }, x) |v, *p| p.* = v;

    const y: []f64 = try VecF64.alloc(3);
    inline for (.{ -1.0, 2.0, 1.0 }, y) |v, *p| p.* = v;

    const z: []f64 = try VecF64.alloc(3);

    defer {
        MatF64.free(A, 4, 3);
        VecF64.free(x);
        VecF64.free(y);
        VecF64.free(z);
    }

    const a_list: [4]f64 = .{ 0.0, 1.0, -1.0, 3.0 };
    const b_list: [4]f64 = .{ 0.0, 1.0, -1.0, -5.0 };
    const z_list: [16][3]f64 = .{
        .{ 0.0, 0.0, 0.0 },
        .{ -1.0, 2.0, 1.0 },
        .{ 1.0, -2.0, -1.0 },
        .{ 5.0, -10.0, -5.0 },
        .{ 36.0, -145.0, 3.0 },
        .{ 35.0, -143.0, 4.0 },
        .{ 37.0, -147.0, 2.0 },
        .{ 41.0, -155.0, -2.0 },
        .{ -36.0, 145.0, -3.0 },
        .{ -37.0, 147.0, -2.0 },
        .{ -35.0, 143.0, -4.0 },
        .{ -31.0, 135.0, -8.0 },
        .{ 108.0, -435.0, 9.0 },
        .{ 107.0, -433.0, 10.0 },
        .{ 109.0, -437.0, 8.0 },
        .{ 113.0, -445.0, 4.0 },
    };

    for (a_list, 0..) |a, ia| {
        for (b_list, 0..) |b, ib| {
            gemv(.T, a, A, x, b, y, z);

            try testing.expect(std.mem.eql(f64, &z_list[4 * ia + ib], z));
        }
    }
}

pub fn geru(comptime mode: WriteMode, a: f64, x: []f64, y: []f64, A: [][]f64) void {
    switch (mode) {
        .W => {
            if (a == 0.0) {
                for (A) |row| {
                    for (row) |*pr| pr.* = 0.0;
                }
            } else if (a == 1.0) {
                for (A, x) |row, vx| {
                    if (vx != 0.0) {
                        for (row, y) |*pr, vy| pr.* = vx * vy;
                    }
                }
            } else if (a == -1.0) {
                var tmp: f64 = undefined;
                for (A, x) |row, vx| {
                    if (vx != 0.0) {
                        tmp = -vx;
                        for (row, y) |*pr, vy| pr.* = tmp * vy;
                    }
                }
            } else {
                var tmp: f64 = undefined;
                for (A, x) |row, vx| {
                    if (vx != 0.0) {
                        tmp = a * vx;
                        for (row, y) |*pr, vy| pr.* = tmp * vy;
                    }
                }
            }
        },
        .U => {
            if (a == 0.0) {
                return;
            } else if (a == 1.0) {
                for (A, x) |row, vx| {
                    if (vx != 0.0) {
                        for (row, y) |*pr, vy| pr.* += vx * vy;
                    }
                }
            } else if (a == -1.0) {
                for (A, x) |row, vx| {
                    if (vx != 0.0) {
                        for (row, y) |*pr, vy| pr.* -= vx * vy;
                    }
                }
            } else {
                var tmp: f64 = undefined;
                for (A, x) |row, vx| {
                    if (vx != 0.0) {
                        tmp = a * vx;
                        for (row, y) |*pr, vy| pr.* += tmp * vy;
                    }
                }
            }
        },
    }

    return;
}

test "blas.geru.W" {
    const MatF64: Matrix = .{ .allocator = testing.allocator };
    const A: [][]f64 = try MatF64.alloc(4, 3);

    const VecF64: Vector = .{ .allocator = testing.allocator };
    const x: []f64 = try VecF64.alloc(4);
    inline for (.{ -1.0, 3.0, -5.0, 1.0 }, x) |v, *p| p.* = v;

    const y: []f64 = try VecF64.alloc(3);
    inline for (.{ 3.0, 1.0, 5.0 }, y) |v, *p| p.* = v;

    defer {
        MatF64.free(A, 4, 3);
        VecF64.free(x);
        VecF64.free(y);
    }

    const a_list: [4]f64 = .{ 0.0, 1.0, -1.0, 3.0 };
    const A_list: [4][4][3]f64 = .{
        .{ .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 } },
        .{ .{ -3.0, -1.0, -5.0 }, .{ 9.0, 3.0, 15.0 }, .{ -15.0, -5.0, -25.0 }, .{ 3.0, 1.0, 5.0 } },
        .{ .{ 3.0, 1.0, 5.0 }, .{ -9.0, -3.0, -15.0 }, .{ 15.0, 5.0, 25.0 }, .{ -3.0, -1.0, -5.0 } },
        .{ .{ -9.0, -3.0, -15.0 }, .{ 27.0, 9.0, 45.0 }, .{ -45.0, -15.0, -75.0 }, .{ 9.0, 3.0, 15.0 } },
    };

    for (a_list, 0..) |a, ia| {
        geru(.W, a, x, y, A);

        for (A_list[ia], A) |ans, row| {
            try testing.expect(std.mem.eql(f64, &ans, row));
        }
    }
}

test "blas.geru.U" {
    const MatF64: Matrix = .{ .allocator = testing.allocator };
    const A: [][]f64 = try MatF64.alloc(4, 3);

    const VecF64: Vector = .{ .allocator = testing.allocator };
    const x: []f64 = try VecF64.alloc(4);
    inline for (.{ -1.0, 3.0, -5.0, 1.0 }, x) |v, *p| p.* = v;

    const y: []f64 = try VecF64.alloc(3);
    inline for (.{ 3.0, 1.0, 5.0 }, y) |v, *p| p.* = v;

    defer {
        MatF64.free(A, 4, 3);
        VecF64.free(x);
        VecF64.free(y);
    }

    const a_list: [4]f64 = .{ 0.0, 1.0, -1.0, 3.0 };
    const A_list: [4][4][3]f64 = .{
        .{ .{ 7.0, -11.0, 0.0 }, .{ 5.0, -17.0, 3.0 }, .{ 1.0, -13.0, 2.0 }, .{ 2.0, -19.0, 0.0 } },
        .{ .{ 4.0, -12.0, -5.0 }, .{ 14.0, -14.0, 18.0 }, .{ -14.0, -18.0, -23.0 }, .{ 5.0, -18.0, 5.0 } },
        .{ .{ 10.0, -10.0, 5.0 }, .{ -4.0, -20.0, -12.0 }, .{ 16.0, -8.0, 27.0 }, .{ -1.0, -20.0, -5.0 } },
        .{ .{ -2.0, -14.0, -15.0 }, .{ 32.0, -8.0, 48.0 }, .{ -44.0, -28.0, -73.0 }, .{ 11.0, -16.0, 15.0 } },
    };

    for (a_list, 0..) |a, ia| {
        inline for (.{ 7.0, -11.0, 0.0 }, A[0]) |v, *p| p.* = v;
        inline for (.{ 5.0, -17.0, 3.0 }, A[1]) |v, *p| p.* = v;
        inline for (.{ 1.0, -13.0, 2.0 }, A[2]) |v, *p| p.* = v;
        inline for (.{ 2.0, -19.0, 0.0 }, A[3]) |v, *p| p.* = v;

        geru(.U, a, x, y, A);

        for (A_list[ia], A) |ans, row| {
            try testing.expect(std.mem.eql(f64, &ans, row));
        }
    }
}
