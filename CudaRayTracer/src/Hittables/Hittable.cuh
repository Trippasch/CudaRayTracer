#pragma once

#include "Core/Log.h"
#include "Hittables/AABB.cuh"
#include "Utils/Ray.cuh"
#include "Utils/helper_cuda.h"

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

class Material;

typedef struct HitRecord
{
    Vec3 p;
    Vec3 normal;
    Material* mat_ptr;
    float t;
    float u, v;
    bool front_face;

    __device__ inline void SetFaceNormal(const Ray& r, const Vec3& outward_normal)
    {
        front_face = Dot(r.Direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
} HitRecord;

typedef enum HittableType
{
    SPHERE,
    XYRECT,
    XZRECT,
    YZRECT,
    HITTABLELIST,
    BVHNODE
} HittableType;

class Sphere;
class XYRect;
class XZRect;
class YZRect;
class HittableList;
class BVHNode;

class Hittable
{
public:
    HittableType type;
    bool isActive;

    union ObjectUnion {
        Sphere* sphere;
        XYRect* xy_rect;
        XZRect* xz_rect;
        YZRect* yz_rect;
        HittableList* hittable_list;
        BVHNode* bvh_node;
    };

    ObjectUnion* Object;

    __host__ Hittable(const Hittable& other) : type(other.type), isActive(other.isActive), Object(other.Object)
    {
    }
};

class Sphere
{
public:
    Vec3 center;
    float radius;
    Material* mat_ptr;

    __host__ Sphere(Vec3 cen, float r, Material* m) : center(cen), radius(r), mat_ptr(m)
    {
    }

    __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        Vec3 oc = r.Origin() - center;
        float a = Dot(r.Direction(), r.Direction());
        float b = Dot(oc, r.Direction());
        float c = Dot(oc, oc) - radius * radius;

        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float temp = (-b - sqrtf(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.PointAtParameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                GetSphereUV(rec.normal, rec.u, rec.v);
                rec.mat_ptr = mat_ptr;
                return true;
            }
            temp = (-b + sqrtf(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.PointAtParameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                GetSphereUV(rec.normal, rec.u, rec.v);
                rec.mat_ptr = mat_ptr;
                return true;
            }
        }

        return false;
    }

    __host__ inline bool BoundingBox(AABB& output_box) const
    {
        output_box = AABB(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius));
        return true;
    }

private:
    __device__ static inline void GetSphereUV(const Vec3& p, float& u, float& v)
    {
        float theta = acos(-p.y());
        float phi = atan2(-p.z(), p.x()) + PI;
        u = phi / (2 * PI);
        v = theta / PI;
    }
};

class XYRect
{
public:
    Vec3 center;
    float width;
    float height;
    Material* mat_ptr;

    __host__ XYRect(Vec3 cen, float w, float h, Material* m) : center(cen), width(w), height(h), mat_ptr(m)
    {
    }

    __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        float x0, x1, y0, y1, k;
        x0 = center.x() - (width / 2);
        x1 = center.x() + (width / 2);
        y0 = center.y() - (height / 2);
        y1 = center.y() + (height / 2);
        k = center.z();

        float inv_dz = 1.0f / r.Direction().z();
        float t = (k - r.Origin().z()) * inv_dz;

        if (t < t_min || t > t_max)
            return false;

        float x = r.Origin().x() + t * r.Direction().x();
        float y = r.Origin().y() + t * r.Direction().y();
        if (x < x0 || x > x1 || y < y0 || y > y1)
            return false;

        rec.u = (x - x0) / (x1 - x0);
        rec.v = (y - y0) / (y1 - y0);
        rec.t = t;
        Vec3 outward_normal = Vec3(0.0f, 0.0f, 1.0f);
        rec.SetFaceNormal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        rec.p = r.PointAtParameter(t);

        return true;
    }

    __host__ inline bool BoundingBox(AABB& output_box) const
    {
        float x0, x1, y0, y1, k;
        x0 = center.x() - (width / 2);
        x1 = center.x() + (width / 2);
        y0 = center.y() - (height / 2);
        y1 = center.y() + (height / 2);
        k = center.z();
        output_box = AABB(Vec3(x0, y0, k - 0.0001f), Vec3(x1, y1, k + 0.0001f));
        return true;
    }
};

class XZRect
{
public:
    Vec3 center;
    float width;
    float height;
    Material* mat_ptr;

    __host__ XZRect(Vec3 cen, float w, float h, Material* m) : center(cen), width(w), height(h), mat_ptr(m)
    {
    }

    __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        float x0, x1, z0, z1, k;
        x0 = center.x() - (width / 2);
        x1 = center.x() + (width / 2);
        z0 = center.z() - (height / 2);
        z1 = center.z() + (height / 2);
        k = center.y();

        float inv_dy = 1.0f / r.Direction().y();
        float t = (k - r.Origin().y()) * inv_dy;

        if (t < t_min || t > t_max)
            return false;

        float x = r.Origin().x() + t * r.Direction().x();
        float z = r.Origin().z() + t * r.Direction().z();
        if (x < x0 || x > x1 || z < z0 || z > z1)
            return false;

        rec.u = (x - x0) / (x1 - x0);
        rec.v = (z - z0) / (z1 - z0);
        rec.t = t;
        Vec3 outward_normal = Vec3(0.0f, 1.0f, 0.0f);
        rec.SetFaceNormal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        rec.p = r.PointAtParameter(t);

        return true;
    }

    __host__ inline bool BoundingBox(AABB& output_box) const
    {
        float x0, x1, z0, z1, k;
        x0 = center.x() - (width / 2);
        x1 = center.x() + (width / 2);
        z0 = center.z() - (height / 2);
        z1 = center.z() + (height / 2);
        k = center.y();
        output_box = AABB(Vec3(x0, k - 0.0001f, z0), Vec3(x1, k + 0.0001f, z1));
        return true;
    }
};

class YZRect
{
public:
    Vec3 center;
    float width;
    float height;
    Material* mat_ptr;

    __host__ YZRect(Vec3 cen, float w, float h, Material* m) : center(cen), width(w), height(h), mat_ptr(m)
    {
    }

    __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        float y0, y1, z0, z1, k;
        y0 = center.y() - (height / 2);
        y1 = center.y() + (height / 2);
        z0 = center.z() - (width / 2);
        z1 = center.z() + (width / 2);
        k = center.x();

        float inv_dx = 1.0f / r.Direction().x();
        float t = (k - r.Origin().x()) * inv_dx;

        if (t < t_min || t > t_max)
            return false;

        float y = r.Origin().y() + t * r.Direction().y();
        float z = r.Origin().z() + t * r.Direction().z();
        if (y < y0 || y > y1 || z < z0 || z > z1)
            return false;

        rec.u = (y - y0) / (y1 - y0);
        rec.v = (z - z0) / (z1 - z0);
        rec.t = t;
        Vec3 outward_normal = Vec3(1.0f, 0.0f, 0.0f);
        rec.SetFaceNormal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        rec.p = r.PointAtParameter(t);

        return true;
    }

    __host__ inline bool BoundingBox(AABB& output_box) const
    {
        float y0, y1, z0, z1, k;
        y0 = center.y() - (height / 2);
        y1 = center.y() + (height / 2);
        z0 = center.z() - (width / 2);
        z1 = center.z() + (width / 2);
        k = center.x();
        output_box = AABB(Vec3(k - 0.0001f, y0, z0), Vec3(k + 0.0001f, y1, z1));
        return true;
    }
};

class BVHNode
{
public:
    AABB box;
    Hittable* left;
    Hittable* right;

    __host__ BVHNode(Hittable** list, size_t start, size_t end)
    {
        Hittable** objects = new Hittable*[end];
        for (int i = start; i < end; i++) {
            objects[i] = list[i];
        }

        // First filter out inactive objects from the list
        auto activeObjectsEnd =
            thrust::remove_if(objects + start, objects + end, [](const Hittable* object) { return !object->isActive; });

        size_t object_span = activeObjectsEnd - (objects + start);

        // Check if there are no active objects in this subset
        if (object_span == 0) {
            left = right = nullptr;
            return;
        }

        // Now proceed with the original constructor
        thrust::sort(objects + start, activeObjectsEnd,
                     [](const Hittable* a, const Hittable* b) { return a->type < b->type; });

        if (object_span == 1) {
            left = right = objects[start];
        }
        else if (object_span == 2) {
            left = objects[start];
            right = objects[start + 1];
        }
        else {
            auto mid = thrust::partition(
                objects + start, activeObjectsEnd,
                [type = objects[start]->type](const Hittable* object) { return object->type == type; });

            size_t mid_index = mid - objects;

            if (mid_index == start || mid_index == activeObjectsEnd - objects) {
                mid_index = start + object_span / 2;
            }

            // Calculate total size of memory needed
            size_t totalSize = 2 * (sizeof(Hittable) + sizeof(Hittable::ObjectUnion) + sizeof(BVHNode));

            // Allocate the memory
            checkCudaErrors(cudaMallocManaged(&memory, totalSize));

            // Partition the memory
            left = (Hittable*)memory;
            left->Object = (Hittable::ObjectUnion*)(left + 1);
            left->Object->bvh_node = (BVHNode*)(left->Object + 1);
            left->type = HittableType::BVHNODE;

            right = (Hittable*)(left->Object->bvh_node + 1);
            right->Object = (Hittable::ObjectUnion*)(right + 1);
            right->Object->bvh_node = (BVHNode*)(right->Object + 1);
            right->type = HittableType::BVHNODE;

            left->Object->bvh_node = new (left->Object->bvh_node) BVHNode(objects, start, mid_index);
            right->Object->bvh_node =
                new (right->Object->bvh_node) BVHNode(objects, mid_index, activeObjectsEnd - objects);
        }

        AABB box_left, box_right;

        bool b_l = left && (left->type == HittableType::SPHERE   ? left->Object->sphere->BoundingBox(box_left)
                            : left->type == HittableType::XYRECT ? left->Object->xy_rect->BoundingBox(box_left)
                            : left->type == HittableType::XZRECT ? left->Object->xz_rect->BoundingBox(box_left)
                            : left->type == HittableType::YZRECT ? left->Object->yz_rect->BoundingBox(box_left)
                                                                 : left->Object->bvh_node->BoundingBox(box_left));

        bool b_r = right && (right->type == HittableType::SPHERE   ? right->Object->sphere->BoundingBox(box_right)
                             : right->type == HittableType::XYRECT ? right->Object->xy_rect->BoundingBox(box_right)
                             : right->type == HittableType::XZRECT ? right->Object->xz_rect->BoundingBox(box_right)
                             : right->type == HittableType::YZRECT ? right->Object->yz_rect->BoundingBox(box_right)
                                                                   : right->Object->bvh_node->BoundingBox(box_right));

        if (!b_l || !b_r) {
            RT_ERROR("No bounding box in bvh_node constructor.");
        }

        box = SurroundingBox(box_left, box_right);
    }

    __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        if (!box.Hit(r, t_min, t_max)) {
            return false;
        }

        struct StackNode
        {
            const BVHNode* node;
            float t_min, t_max;
        };

        StackNode stack[2]; // Reduced stack size cause of register spilling
        int top = -1;

        stack[++top] = {this, t_min, t_max};

        bool hit_something = false;
        StackNode current;
        const Hittable* obj;
        Hittable::ObjectUnion* obj_object;

        while (top >= 0) {
            current = stack[top--];

            if (!current.node->box.Hit(r, current.t_min, current.t_max)) {
                continue;
            }

            const Hittable* left = current.node->left;
            const Hittable* right = current.node->right;

            if (left && left->type == BVHNODE) {
                obj = left;
                obj_object = obj->Object;
                stack[++top] = {obj_object->bvh_node, current.t_min, hit_something ? rec.t : current.t_max};
            }
            else if (left) {
                hit_something |= PerformHit(r, current.t_min, hit_something ? rec.t : current.t_max, rec, left);
            }

            if (right && right->type == BVHNODE) {
                obj = right;
                obj_object = obj->Object;
                stack[++top] = {obj_object->bvh_node, current.t_min, hit_something ? rec.t : current.t_max};
            }
            else if (right) {
                hit_something |= PerformHit(r, current.t_min, hit_something ? rec.t : current.t_max, rec, right);
            }
        }

        return hit_something;
    }

    __host__ inline bool BoundingBox(AABB& output_box) const
    {
        output_box = box;
        return true;
    }

    __host__ inline void Destroy()
    {
        bool leftIsNode = false;
        bool rightIsNode = false;

        if (left && left->type == HittableType::BVHNODE) {
            left->Object->bvh_node->Destroy();
            leftIsNode = true;
        }

        if (right && right->type == HittableType::BVHNODE) {
            right->Object->bvh_node->Destroy();
            rightIsNode = true;
        }

        if (leftIsNode || rightIsNode) {
            checkCudaErrors(cudaFree(memory));
        }
    }

private:
    char* memory;

    __device__ inline bool PerformHit(const Ray& r, float t_min, float t_max, HitRecord& rec, const Hittable* obj) const
    {
        Hittable::ObjectUnion* obj_object = obj->Object;
        switch (obj->type) {
        case HittableType::SPHERE:
            return obj_object->sphere->Hit(r, t_min, t_max, rec);
        case HittableType::XYRECT:
            return obj_object->xy_rect->Hit(r, t_min, t_max, rec);
        case HittableType::XZRECT:
            return obj_object->xz_rect->Hit(r, t_min, t_max, rec);
        case HittableType::YZRECT:
            return obj_object->yz_rect->Hit(r, t_min, t_max, rec);
        default:
            return false;
        }
    }

    __host__ static inline bool BoxCompare(const Hittable* a, const Hittable* b, int axis)
    {
        AABB box_a;
        AABB box_b;

        bool b_a, b_b;
        if (a->type == HittableType::SPHERE || b->type == HittableType::SPHERE) {
            b_a = a->Object->sphere->BoundingBox(box_a);
            b_b = b->Object->sphere->BoundingBox(box_b);
        }
        else if (a->type == HittableType::XYRECT || b->type == HittableType::XYRECT) {
            b_a = a->Object->xy_rect->BoundingBox(box_a);
            b_b = b->Object->xy_rect->BoundingBox(box_b);
        }
        else if (a->type == HittableType::XZRECT || b->type == HittableType::XZRECT) {
            b_a = a->Object->xz_rect->BoundingBox(box_a);
            b_b = b->Object->xz_rect->BoundingBox(box_b);
        }
        else if (a->type == HittableType::YZRECT || b->type == HittableType::YZRECT) {
            b_a = a->Object->yz_rect->BoundingBox(box_a);
            b_b = b->Object->yz_rect->BoundingBox(box_b);
        }

        if (!b_a || !b_b)
            RT_ERROR("No bounding box in bvh_node constructor.");

        return box_a.Min().e[axis] < box_b.Min().e[axis];
    }

    __host__ static inline bool BoxXCompare(const Hittable* a, const Hittable* b)
    {
        return BoxCompare(a, b, 0);
    }

    __host__ static inline bool BoxYCompare(const Hittable* a, const Hittable* b)
    {
        return BoxCompare(a, b, 1);
    }

    __host__ static inline bool BoxZCompare(const Hittable* a, const Hittable* b)
    {
        return BoxCompare(a, b, 2);
    }
};

class HittableList
{
public:
    Hittable** list;
    int list_size;

    __host__ HittableList()
    {
    }
    __host__ HittableList(Hittable** l, int n)
    {
        list = l;
        list_size = n;
    }

    __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (int i = 0; i < list_size; i++) {
            if (list[i]->Object->bvh_node->Hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __host__ inline bool BoundingBox(AABB& output_box) const
    {
        if (list_size == 0)
            return false;

        AABB temp_box;
        bool first_box = true;

        for (size_t i = 0; i < list_size; i++) {
            if (!list[i]->Object->bvh_node->BoundingBox(temp_box))
                return false;
            output_box = first_box ? temp_box : SurroundingBox(output_box, temp_box);
            first_box = false;
        }

        return true;
    }
};

__forceinline__ __host__ const char* GetTextForEnum(int enumVal)
{
    switch (enumVal) {
    case HittableType::SPHERE:
        return "Sphere ";
    case HittableType::XYRECT:
        return "XY Rectangle ";
    case HittableType::XZRECT:
        return "XZ Rectangle ";
    case HittableType::YZRECT:
        return "YZ Rectangle ";
    case HittableType::HITTABLELIST:
        return "Hittable List ";
    case HittableType::BVHNODE:
        return "BVH Node ";

    default:
        return "Not recognized.. ";
    }
}