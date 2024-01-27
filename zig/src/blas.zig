const std = @import("std");
const testing = std.testing;

pub fn copy(comptime T: type, des: []T, src: []T) void {
    if (@typeInfo(T) != .Float) {
        @compileError("copy: T should be a float type.");
    }
    for (des, src) |*des_ptr, src_val| {
        des_ptr.* = src_val;
    }
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

    for (src, 0..) |*ptr, ind| {
        ptr.* = @floatFromInt(ind);
    }

    copy(f32, des, src);
    for (des, src) |des_val, src_val| {
        try testing.expectEqual(des_val, src_val);
    }
}

pub fn xoty(comptime T: type, x: []T, y: []T, z: []T) void {
    if (@typeInfo(T) != .Float) {
        @compileError("xoty: T should be a float type.");
    }
    for (x, y, z) |val_x, val_y, *ptr_z| {
        ptr_z.* = val_x * val_y;
    }
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

    for (x, y, 0..) |*ptr_x, *ptr_y, ind| {
        ptr_x.* = @floatFromInt(2 * ind);
        ptr_y.* = @floatFromInt(3 * ind);
    }

    xoty(f32, x, y, z);
    for (z, 0..) |val, ind| {
        try testing.expectEqual(val, @as(f32, @floatFromInt(6 * ind * ind)));
    }
}
