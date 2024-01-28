const std = @import("std");
const testing = std.testing;

pub fn copy(comptime T: type, des: []T, src: []T) void {
    if (@typeInfo(T) != .Float) @compileError("copy(T, ...): T should be a float type.");
    for (des, src) |*pd, vs| pd.* = vs;
    return;
}

test "blas.copy" {
    const allocator = testing.allocator;
    const src: []f32 = try allocator.alloc(f32, 10);
    const des: []f32 = try allocator.alloc(f32, 10);
    defer {
        allocator.free(src);
        allocator.free(des);
    }

    for (src, 0..) |*p, i| p.* = @floatFromInt(i);

    copy(f32, des, src);
    for (des, src) |vd, vs| try testing.expectEqual(vd, vs);
}

pub fn xoty(comptime T: type, x: []T, y: []T, z: []T) void {
    if (@typeInfo(T) != .Float) @compileError("xoty(T, ...): T should be a float type.");
    for (x, y, z) |vx, vy, *pz| pz.* = vx * vy;
    return;
}

test "blas.xoty" {
    const allocator = testing.allocator;
    const x: []f32 = try allocator.alloc(f32, 10);
    const y: []f32 = try allocator.alloc(f32, 10);
    const z: []f32 = try allocator.alloc(f32, 10);
    defer {
        allocator.free(x);
        allocator.free(y);
        allocator.free(z);
    }

    for (x, y, 0..) |*px, *py, i| {
        px.* = @floatFromInt(2 * i);
        py.* = @floatFromInt(3 * i);
    }

    xoty(f32, x, y, z);
    for (z, 0..) |v, i| try testing.expectEqual(v, @as(f32, @floatFromInt(6 * i * i)));
}

const Transpose = enum(u1) { N, T };

pub fn gemv(comptime T: type, comptime tA: Transpose, a: T, A: [][]T, x: []T, b: T, y: []T, z: []T) void {
    if (@typeInfo(T) != .Float) @compileError("gemv(T, ...): T should be a float type.");

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
            var tmp: T = undefined;
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
                var tmp: T = undefined;
                for (A, x) |row, vx| {
                    tmp = a * vx;
                    for (row, z) |vr, *pz| pz.* += tmp * vr;
                }
            }
        },
    }

    return;
}

test "gemv.N" {
    const allocator = testing.allocator;
    const A: [][]f32 = blk: {
        const mat: [][]f32 = try allocator.alloc([]f32, 4);
        for (mat) |*row| row.* = try allocator.alloc(f32, 3);
        break :blk mat;
    };
    inline for (.{ 4.0, 3.0, 1.0 }, A[0]) |v, *p| p.* = v;
    inline for (.{ 3.0, 7.0, 0.0 }, A[1]) |v, *p| p.* = v;
    inline for (.{ 2.0, 5.0, 3.0 }, A[2]) |v, *p| p.* = v;
    inline for (.{ 1.0, 1.0, 2.0 }, A[3]) |v, *p| p.* = v;

    const x: []f32 = try allocator.alloc(f32, 3);
    inline for (.{ 3.0, 1.0, 5.0 }, x) |v, *p| p.* = v;

    const y: []f32 = try allocator.alloc(f32, 4);
    inline for (.{ -1.0, 3.0, -5.0, 1.0 }, y) |v, *p| p.* = v;

    const z: []f32 = try allocator.alloc(f32, 4);

    defer {
        for (A) |row| allocator.free(row);
        allocator.free(A);
        allocator.free(x);
        allocator.free(y);
        allocator.free(z);
    }

    const a_list: [3]f32 = .{ 0.0, 1.0, 3.0 };
    const b_list: [4]f32 = .{ 0.0, 1.0, -1.0, -5.0 };
    const z_list: [12][4]f32 = .{
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ -1.0, 3.0, -5.0, 1.0 },
        .{ 1.0, -3.0, 5.0, -1.0 },
        .{ 5.0, -15.0, 25.0, -5.0 },
        .{ 20.0, 16.0, 26.0, 14.0 },
        .{ 19.0, 19.0, 21.0, 15.0 },
        .{ 21.0, 13.0, 31.0, 13.0 },
        .{ 25.0, 1.0, 51.0, 9.0 },
        .{ 60.0, 48.0, 78.0, 42.0 },
        .{ 59.0, 51.0, 73.0, 43.0 },
        .{ 61.0, 45.0, 83.0, 41.0 },
        .{ 65.0, 33.0, 103.0, 37.0 },
    };

    for (a_list, 0..) |a, ia| {
        for (b_list, 0..) |b, ib| {
            gemv(f32, .N, a, A, x, b, y, z);

            for (z_list[4 * ia + ib], z) |va, vz| {
                try testing.expectEqual(va, vz);
            }
        }
    }
}

test "gemv.T" {
    const allocator = testing.allocator;
    const A: [][]f32 = blk: {
        const mat: [][]f32 = try allocator.alloc([]f32, 4);
        for (mat) |*row| row.* = try allocator.alloc(f32, 3);
        break :blk mat;
    };
    inline for (.{ 7.0, -11.0, 0.0 }, A[0]) |v, *p| p.* = v;
    inline for (.{ 5.0, -17.0, 3.0 }, A[1]) |v, *p| p.* = v;
    inline for (.{ 1.0, -13.0, 2.0 }, A[2]) |v, *p| p.* = v;
    inline for (.{ 2.0, -19.0, 0.0 }, A[3]) |v, *p| p.* = v;

    const x: []f32 = try allocator.alloc(f32, 4);
    inline for (.{ 3.0, 1.0, 0.0, 5.0 }, x) |v, *p| p.* = v;

    const y: []f32 = try allocator.alloc(f32, 3);
    inline for (.{ -1.0, 2.0, 1.0 }, y) |v, *p| p.* = v;

    const z: []f32 = try allocator.alloc(f32, 3);

    defer {
        for (A) |row| allocator.free(row);
        allocator.free(A);
        allocator.free(x);
        allocator.free(y);
        allocator.free(z);
    }

    const a_list: [3]f32 = .{ 0.0, 1.0, 3.0 };
    const b_list: [4]f32 = .{ 0.0, 1.0, -1.0, -5.0 };
    const z_list: [12][3]f32 = .{
        .{ 0.0, 0.0, 0.0 },
        .{ -1.0, 2.0, 1.0 },
        .{ 1.0, -2.0, -1.0 },
        .{ 5.0, -10.0, -5.0 },
        .{ 36.0, -145.0, 3.0 },
        .{ 35.0, -143.0, 4.0 },
        .{ 37.0, -147.0, 2.0 },
        .{ 41.0, -155.0, -2.0 },
        .{ 108.0, -435.0, 9.0 },
        .{ 107.0, -433.0, 10.0 },
        .{ 109.0, -437.0, 8.0 },
        .{ 113.0, -445.0, 4.0 },
    };

    for (a_list, 0..) |a, ia| {
        for (b_list, 0..) |b, ib| {
            gemv(f32, .T, a, A, x, b, y, z);

            for (z_list[4 * ia + ib], z) |va, vz| {
                try testing.expectEqual(va, vz);
            }
        }
    }
}
