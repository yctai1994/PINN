const std = @import("std");
const testing = std.testing;

pub fn copy(comptime T: type, des: []T, src: []T) void {
    if (@typeInfo(T) != .Float) {
        @compileError("copy: T should be a float type.");
    }
    for (des, src) |*ptr_des, *ptr_src| {
        ptr_des.* = ptr_src.*;
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
    for (des, src) |*ptr_des, *ptr_src| {
        try testing.expectEqual(ptr_des.*, ptr_src.*);
    }
}
