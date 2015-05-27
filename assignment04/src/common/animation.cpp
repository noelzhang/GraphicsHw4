#include "animation.h"
#include "tesselation.h"

// compute the frame from an animation
frame3f animate_compute_frame(FrameAnimation* animation, int time) {
    // grab keyframe interval
    auto interval = 0;
    for(auto t : animation->keytimes) if(time < t) break; else interval++;
    interval--;
    // get translation and rotation matrices
    auto t = float(time-animation->keytimes[interval])/float(animation->keytimes[interval+1]-animation->keytimes[interval]);
    auto m_t = translation_matrix(animation->translation[interval]*(1-t)+animation->translation[interval+1]*t);
    auto m_rz = rotation_matrix(animation->rotation[interval].z*(1-t)+animation->rotation[interval+1].z*t,z3f);
    auto m_ry = rotation_matrix(animation->rotation[interval].y*(1-t)+animation->rotation[interval+1].y*t,y3f);
    auto m_rx = rotation_matrix(animation->rotation[interval].x*(1-t)+animation->rotation[interval+1].x*t,x3f);
    // compute combined xform matrix
    auto m = m_t * m_rz * m_ry * m_rx;
    // return the transformed frame
    return transform_frame(m, animation->rest_frame);
}

// update mesh frames for animation
void animate_frame(Scene* scene) {
    // foreach mesh
    for(auto mesh : scene->meshes) {
        // if not animation, continue
        if(not mesh->animation) continue;
        // update frame
        mesh->frame = animate_compute_frame(mesh->animation, scene->animation->time);
    }
    // foreach surface
    for(auto surface : scene->surfaces) {
        // if not animation, continue
        if(not surface->animation) continue;
        // update frame
        surface->frame = animate_compute_frame(surface->animation, scene->animation->time);
        // update the _display_mesh
        if(surface->_display_mesh) surface->_display_mesh->frame = surface->frame;
    }
}

// skinning scene
void animate_skin(Scene* scene) {
    // foreach mesh
    for(auto mesh : scene->meshes) {
        // if no skinning, continue
        if(not mesh->skinning) continue;
        // foreach vertex index
        for(auto i : range(mesh->pos.size())) {
            // set pos/norm to zero
            mesh->pos[i] = zero3f;
            mesh->norm[i] = zero3f;
            // for each bone slot (0..3)
            for(auto j : range(4)) {
                // get bone weight and index
                auto w = mesh->skinning->vert_bone_weights[i][j];
                auto bi = mesh->skinning->vert_bone_ids[i][j];
                // if index < 0, continue
                if(bi < 0) continue;
                // grab bone xform
                auto xf = mesh->skinning->bone_xforms[scene->animation->time][bi];
                // update position and normal
                mesh->pos[i] += w * transform_point(xf,mesh->skinning->vert_rest_pos[i]);
                mesh->norm[i] += w * transform_normal(xf,mesh->skinning->vert_rest_norm[i]);
            }
            // normalize normal
            mesh->norm[i] = normalize(mesh->norm[i]);
        }
    }
}

// particle simulation
void simulate(Scene* scene) {
    // for each mesh
    for(auto mesh : scene->meshes) {
        // skip if no simulation
        if(not mesh->simulation) continue;
        // compute time per step
        auto ddt = scene->animation->dt / scene->animation->simsteps;
        // foreach simulation steps
        for(auto j : range(scene->animation->simsteps)) {
            // compute extenal forces (gravity)
            for(auto i : range(mesh->simulation->force.size())) mesh->simulation->force[i] = scene->animation->gravity * mesh->simulation->mass[i];
            // for each spring, compute spring force on points
            for(auto spring : mesh->simulation->springs) {
                // compute spring distance and length
                auto delta_pos = mesh->pos[spring.ids.y] - mesh->pos[spring.ids.x];
                auto spring_dir = normalize(delta_pos);
                auto spring_length = length(delta_pos);
                // compute static force
                auto fs = spring_dir * (spring_length - spring.restlength) * spring.ks;
                // accumulate static force on points
                mesh->simulation->force[spring.ids.x] +=  fs;
                mesh->simulation->force[spring.ids.y] += -fs;
                // compute dynamic force
                auto delta_vel = mesh->simulation->vel[spring.ids.y] - mesh->simulation->vel[spring.ids.x];
                // accumulate dynamic force on points
                auto fd = dot(delta_vel,spring_dir) * spring_dir * spring.kd;
                mesh->simulation->force[spring.ids.x] +=  fd;
                mesh->simulation->force[spring.ids.y] += -fd;
            }
            // newton laws
            for(auto i : range(mesh->pos.size())) {
                // if pinned, skip
                if(mesh->simulation->pinned[i]) continue;
                // acceleration
                auto acc = mesh->simulation->force[i] / mesh->simulation->mass[i];
                // update velocity and positions using Euler's method
                mesh->pos[i] += ddt * mesh->simulation->vel[i] + ddt * ddt * acc / 2;
                mesh->simulation->vel[i] += ddt * acc;
                // for each mesh, check for collision
                for(auto collider : scene->surfaces) {
                    // compute inside tests
                    auto inside = false; auto pos = zero3f, norm = zero3f;
                    // if quad
                    if(collider->isquad) {
                        // compute local poisition
                        auto lpos = transform_point_inverse(collider->frame, mesh->pos[i]);
                        // perform inside test
                        if(lpos.z < 0 and lpos.x > -collider->radius and lpos.x < collider->radius
                           and lpos.y > -collider->radius and lpos.y < collider->radius) {
                            // if inside, set position and normal
                            inside = true;
                            pos = transform_point(collider->frame, {lpos.x,lpos.y,0});
                            norm = collider->frame.z;
                        }
                        // else sphere
                    } else {
                        // inside test
                        auto rr = (mesh->pos[i]-collider->frame.o) / collider->radius;
                        if(length(rr) < 1) {
                            // if inside, set position and normal
                            inside = true;
                            pos = collider->frame.o + (collider->radius) * normalize(rr);
                            norm = normalize(rr);
                        }
                    }
                    // if inside
                    if(inside) {
                        // set particle position
                        mesh->pos[i] = pos;
                        // update velocity
                        auto vel = mesh->simulation->vel[i];
                        mesh->simulation->vel[i] = (vel - dot(norm,vel)*norm)*(1-scene->animation->bounce_dump.x) -
                        dot(norm, vel)*norm*(1-scene->animation->bounce_dump.y);
                    }
                }
            }
        }
        // smooth normals if it has triangles or quads
        if(not mesh->triangle.empty() or not mesh->quad.empty()) smooth_normals(mesh);
    }
}

// scene reset
void animate_reset(Scene* scene) {
    scene->animation->time = 0;
    for(auto mesh : scene->meshes) {
        if(mesh->animation) {
            mesh->frame = mesh->animation->rest_frame;
        }
        if(mesh->skinning) {
            mesh->pos = mesh->skinning->vert_rest_pos;
            mesh->norm = mesh->skinning->vert_rest_norm;
        }
        if(mesh->simulation) {
            mesh->pos = mesh->simulation->init_pos;
            mesh->simulation->vel = mesh->simulation->init_vel;
            mesh->simulation->force.resize(mesh->simulation->init_pos.size());
        }
    }
}

// scene update
void animate_update(Scene* scene) {
    scene->animation->time ++;
    if(scene->animation->time >= scene->animation->length) animate_reset(scene);
    animate_frame(scene);
    animate_skin(scene);
    simulate(scene);
}


