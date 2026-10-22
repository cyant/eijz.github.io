/* ============================================================================
 * dice-physics.js — 无依赖的 D20 刚体模拟
 *
 * 思路：给一次投掷的初始状态（位置 / 姿态 / 线速度 / 角速度），用固定步长把
 * 整个「起飞 → 弹跳 → 翻滚 → 减速 → 停稳」跑完，返回一条可以逐帧回放的轨迹。
 *
 * 碰撞：12 个顶点 vs 无限平面（桌面 y = 0）
 *      冲量法（法向 + 库仑摩擦，累积冲量钳制），位置补偿，接触阻尼，休眠判定
 * 惯量：正多面体关于中心的惯量张量是各向同性的，所以用标量 I 即可
 *
 * 不依赖 three.js —— 方便单独用 Node 跑参数回归。
 * ========================================================================== */

const PHI = (1 + Math.sqrt(5)) / 2;

/* ----------------------------------------------------------------------------
 * 生成骰子几何：12 个顶点（外接球半径 1）、20 个三角面、对面的点数和为 21
 * -------------------------------------------------------------------------- */
export function createDie(){
  const raw = [];
  for (const a of [-1, 1]) for (const b of [-1, 1]){
    raw.push([0, a, b * PHI]);
    raw.push([a, b * PHI, 0]);
    raw.push([a * PHI, 0, b]);
  }
  for (const v of raw){
    const l = Math.hypot(v[0], v[1], v[2]);
    v[0] /= l; v[1] /= l; v[2] /= l;
  }

  const dist = (i, j) => Math.hypot(
    raw[i][0] - raw[j][0], raw[i][1] - raw[j][1], raw[i][2] - raw[j][2]
  );

  // 棱长 = 所有顶点对里的最小距离，三条边都等于它的三点组就是面
  let edge = Infinity;
  for (let i = 0; i < 12; i++) for (let j = i + 1; j < 12; j++) edge = Math.min(edge, dist(i, j));
  const tol = edge * 0.02;

  const faces = [];
  for (let i = 0; i < 12; i++) for (let j = i + 1; j < 12; j++) for (let k = j + 1; k < 12; k++){
    if (Math.abs(dist(i, j) - edge) > tol) continue;
    if (Math.abs(dist(j, k) - edge) > tol) continue;
    if (Math.abs(dist(k, i) - edge) > tol) continue;

    let A = raw[i].slice(), B = raw[j].slice(), C = raw[k].slice();
    // 绕序统一成「法线朝外」
    let n = cross(sub(B, A), sub(C, A));
    const c = [(A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3, (A[2] + B[2] + C[2]) / 3];
    if (dot(n, c) < 0){
      const t = B; B = C; C = t;
      n = cross(sub(B, A), sub(C, A));
    }
    n = normalize(n);
    faces.push({
      a: A, b: B, c: C,
      center: c,
      normal: n,
      value: 0
    });
  }

  // 对面：法线反向 → 点数和 21（真实 D20 的排法）
  {
    const used = new Array(faces.length).fill(false);
    let next = 1;
    for (let i = 0; i < faces.length; i++){
      if (used[i]) continue;
      let mate = -1;
      for (let k = i + 1; k < faces.length; k++){
        if (!used[k] && dot(faces[k].normal, faces[i].normal) < -0.99){ mate = k; break; }
      }
      faces[i].value = next;
      used[i] = true;
      if (mate >= 0){ faces[mate].value = 21 - next; used[mate] = true; }
      next++;
    }
  }

  const verts = new Float64Array(36);
  raw.forEach((v, i) => { verts[i * 3] = v[0]; verts[i * 3 + 1] = v[1]; verts[i * 3 + 2] = v[2]; });

  // 内切球半径 = 停稳时球心离桌面的高度
  let inradius = Infinity;
  for (const f of faces) inradius = Math.min(inradius, dot(f.center, f.normal));

  return {
    verts,          // Float64Array(36)：12 个局部顶点
    faces,          // 20 个面：线段端点 / 中心 / 外法线 / 点数
    edge,
    inradius
  };
}

/* ----------------------------------------------------------------------------
 * 当前朝上的面
 * -------------------------------------------------------------------------- */
export function topFaceIndex(die, q){
  let best = 0, bestY = -Infinity;
  for (let i = 0; i < die.faces.length; i++){
    const y = rotateVecY(die.faces[i].normal, q);
    if (y > bestY){ bestY = y; best = i; }
  }
  return best;
}

/* 把向量 n 用四元数 q 旋转后取 y 分量 */
export function rotateVecY(n, q){
  const tx = 2 * (q[1] * n[2] - q[2] * n[1]);
  const ty = 2 * (q[2] * n[0] - q[0] * n[2]);
  const tz = 2 * (q[0] * n[1] - q[1] * n[0]);
  return n[1] + q[3] * ty + (q[2] * tx - q[0] * tz);
}

/* ----------------------------------------------------------------------------
 * 一次投掷的完整模拟
 *
 * init: { pos:[x,y,z], quat:[x,y,z,w], vel:[vx,vy,vz], omega:[wx,wy,wz] }
 * 返回: {
 *   pos, quat,                    // 停稳后的状态
 *   tracks: { n, dt, pos, quat }, // 逐帧轨迹（Float32Array），第 k 帧在 t = k*dt
 *   impacts: [{ t, v }],          // 落地撞击（用来配音效）
 *   duration,                     // 轨迹总时长（秒）
 *   asleep,                       // 是否自然睡着（false = 超时强制停）
 *   topIndex, value, topDot,
 *   maxSpeed, maxHeight
 * }
 * -------------------------------------------------------------------------- */
export function simulateThrow(die, init, opt = {}){
  const o = Object.assign({
    dt: 1 / 240,        // 模拟步长
    gravity: 20,        // 重力加速度（世界单位；骰子外接球半径 1）—— 小一点，翻滚更从容
    restitution: 0.50,  // 恢复系数（金属骰子砸在硬桌面上会弹两下）
    friction: 0.46,     // 库仑摩擦
    inertia: 0.30,      // 单位质量、半径 1 的实心二十面体
    linDamp: 0.05,      // 空气阻力（很弱）
    angDamp: 0.06,
    contactDamp: 1.6,   // 接触时的滚动阻力
    iterations: 8,      // 每步冲量迭代次数
    slop: 0.0015,       // 允许的穿透深度
    correction: 0.75,   // 位置补偿比例
    sleepLin: 0.055,
    sleepAng: 1.0,
    sleepTime: 0.13,
    sleepMinContacts: 3, // 少于 3 个支点不许睡（防止在棱上「金鸡独立」）
    maxTime: 5.0,
    softStop: 2.4,      // 超过这个时间开始「加大摩擦」，保证能停下来
    record: true
  }, opt);

  const dt = o.dt;
  const invI = 1 / o.inertia;
  const V = die.verts;
  const rnd = o.rnd || Math.random;

  // ---- 刚体状态（局部变量，热循环里更好优化）----
  let px = init.pos[0], py = init.pos[1], pz = init.pos[2];
  let qx = init.quat[0], qy = init.quat[1], qz = init.quat[2], qw = init.quat[3];
  let vx = init.vel[0], vy = init.vel[1], vz = init.vel[2];
  let wx = init.omega[0], wy = init.omega[1], wz = init.omega[2];
  {
    const l = Math.hypot(qx, qy, qz, qw) || 1;
    qx /= l; qy /= l; qz /= l; qw /= l;
  }

  // ---- 顶点缓存 / 接触缓存 ----
  const rwx = new Float64Array(12), rwy = new Float64Array(12), rwz = new Float64Array(12);
  const MAXC = 12;
  const crx = new Float64Array(MAXC), cry = new Float64Array(MAXC), crz = new Float64Array(MAXC);
  const cjn = new Float64Array(MAXC), cjt = new Float64Array(MAXC);

  const maxSteps = Math.ceil(o.maxTime / dt) + 2;
  const trackPos = o.record ? new Float32Array(maxSteps * 3) : null;
  const trackQuat = o.record ? new Float32Array(maxSteps * 4) : null;
  let n = 0;

  const impacts = [];
  let sleepT = 0, teeterT = 0, nudgeCool = 0, nudges = 0, t = 0, asleep = false;
  let maxSpeed = 0, maxHeight = py;

  const pushFrame = () => {
    if (!o.record) return;
    trackPos[n * 3] = px; trackPos[n * 3 + 1] = py; trackPos[n * 3 + 2] = pz;
    trackQuat[n * 4] = qx; trackQuat[n * 4 + 1] = qy; trackQuat[n * 4 + 2] = qz; trackQuat[n * 4 + 3] = qw;
    n++;
  };

  pushFrame();

  for (let step = 0; step < maxSteps; step++){
    t += dt;

    // ---- 重力 + 阻尼 ----
    vy -= o.gravity * dt;
    {
      const ld = Math.max(0, 1 - o.linDamp * dt);
      vx *= ld; vy *= ld; vz *= ld;
      const ad = Math.max(0, 1 - o.angDamp * dt);
      wx *= ad; wy *= ad; wz *= ad;
    }
    if (t > o.softStop){
      // 超时辅助：只额外消耗能量，不改变方向 —— 视觉上看不出来，但保证能停
      const sd = Math.max(0, 1 - 4.0 * dt);
      wx *= sd; wy *= sd; wz *= sd;
      vx *= sd; vz *= sd;
    }

    // ---- 顶点转到世界系（相对球心）----
    for (let i = 0, o3 = 0; i < 12; i++, o3 += 3){
      const ix = V[o3], iy = V[o3 + 1], iz = V[o3 + 2];
      const tx = 2 * (qy * iz - qz * iy);
      const ty = 2 * (qz * ix - qx * iz);
      const tz = 2 * (qx * iy - qy * ix);
      rwx[i] = ix + qw * tx + (qy * tz - qz * ty);
      rwy[i] = iy + qw * ty + (qz * tx - qx * tz);
      rwz[i] = iz + qw * tz + (qx * ty - qy * tx);
    }

    // ---- 收集接触点（穿进桌面）----
    let nc = 0, maxDepth = 0, hitSpeed = 0;
    for (let i = 0; i < 12; i++){
      const depth = -(py + rwy[i]);
      if (depth <= 0) continue;
      if (nc === MAXC) break;
      crx[nc] = rwx[i]; cry[nc] = rwy[i]; crz[nc] = rwz[i];
      cjn[nc] = 0; cjt[nc] = 0;
      if (depth > maxDepth) maxDepth = depth;
      if (depth >= 0){
        const vn = vy + (wz * rwx[i] - wx * rwz[i]);
        if (vn < 0 && -vn > hitSpeed) hitSpeed = -vn;
      }
      nc++;
    }

    if (hitSpeed > 1.4 && (impacts.length === 0 || t - impacts[impacts.length - 1].t > 0.05)){
      impacts.push({ t, v: hitSpeed });
    }

    // ---- 顺序冲量求解（法向 + 摩擦）----
    for (let it = 0; it < o.iterations; it++){
      for (let ci = 0; ci < nc; ci++){
        const rx = crx[ci], ry = cry[ci], rz = crz[ci];

        // 法向
        let vn = vy + (wz * rx - wx * rz);
        const e = vn < -1.1 ? o.restitution : 0;
        const denomN = 1 + invI * (rx * rx + rz * rz);
        let dj = -(1 + e) * vn / denomN;
        const oldJn = cjn[ci];
        let jn = oldJn + dj;
        if (jn < 0) jn = 0;
        dj = jn - oldJn;
        cjn[ci] = jn;
        if (dj !== 0){
          vy += dj;
          wx += invI * dj * (-rz);
          wz += invI * dj * (rx);
        }

        // 摩擦（切向单方向约束 + 库仑锥钳制）
        const tvx = vx + (wy * rz - wz * ry);
        const tvz = vz + (wx * ry - wy * rx);
        const vtLen = Math.hypot(tvx, tvz);
        if (vtLen > 1e-5){
          const nx = -tvx / vtLen, nz = -tvz / vtLen;   // 摩擦冲量方向：抵消切向速度
          const ax = ry * nz;
          const ay = rz * nx - rx * nz;
          const az = -ry * nx;
          const denomT = 1 + invI * (ax * ax + ay * ay + az * az);
          let djt = vtLen / denomT;
          const maxF = o.friction * cjn[ci];
          const oldJt = cjt[ci];
          let jt = oldJt + djt;
          if (jt > maxF) jt = maxF;
          else if (jt < -maxF) jt = -maxF;
          djt = jt - oldJt;
          cjt[ci] = jt;
          if (djt !== 0){
            vx += djt * nx;
            vz += djt * nz;
            wx += invI * (ry * (djt * nz) - rz * 0);
            wy += invI * (rz * (djt * nx) - rx * (djt * nz));
            wz += invI * (0 - ry * (djt * nx));
          }
        }
      }
    }

    // ---- 接触阻尼：模拟滚动阻力 ----
    if (nc > 0){
      const cd = Math.max(0, 1 - o.contactDamp * dt);
      wx *= cd; wy *= cd; wz *= cd;
    }

    // ---- 位置补偿（只抬不压）----
    if (maxDepth > o.slop) py += (maxDepth - o.slop) * o.correction;

    // ---- 积分 ----
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    {
      const hx = wx * dt * 0.5, hy = wy * dt * 0.5, hz = wz * dt * 0.5;
      const nx = hx * qw + hy * qz - hz * qy;
      const ny = -hx * qz + hy * qw + hz * qx;
      const nz = hx * qy - hy * qx + hz * qw;
      const nw = -hx * qx - hy * qy - hz * qz;
      qx += nx; qy += ny; qz += nz; qw += nw;
      const l = Math.hypot(qx, qy, qz, qw) || 1;
      qx /= l; qy /= l; qz /= l; qw /= l;
    }

    const sp = Math.hypot(vx, vy, vz);
    if (sp > maxSpeed) maxSpeed = sp;
    if (py > maxHeight) maxHeight = py;

    pushFrame();

    // ---- 在棱/顶点上「金鸡独立」的补偿 ----
    // 骰子只剩 1~2 个支点、又慢吞吞耗了 0.35 秒，说明它卡在「顶点上打转」这种
    // 解析上成立、现实里一瞬间就倒的状态。做法：先卸掉这股像陀螺一样的转速，
    // 再给一记小扰动把它推倒。带冷却，免得反复踢。
    {
      const wlen = Math.hypot(wx, wy, wz);
      if (nc <= 2 && sp < 2.2 && wlen < 3.0) teeterT += dt; else teeterT = 0;
      if (nudgeCool > 0) nudgeCool -= dt;
      if (teeterT > 0.28 && nudgeCool <= 0){
        wx *= 0.30; wy *= 0.30; wz *= 0.30;
        const a = rnd() * Math.PI * 2, k = 0.45 + rnd() * 0.45;
        wx += Math.cos(a) * k;
        wz += Math.sin(a) * k;
        teeterT = 0;
        nudgeCool = 0.45;
        nudges++;
      }
    }

    // ---- 休眠：至少 3 个支点 + 足够慢 + 足够久 ----
    if (nc >= o.sleepMinContacts && sp < o.sleepLin && Math.hypot(wx, wy, wz) < o.sleepAng){
      sleepT += dt;
      if (sleepT > o.sleepTime){
        vx = vy = vz = 0; wx = wy = wz = 0;
        asleep = true;
        break;
      }
    } else {
      sleepT = 0;
    }
  }

  // 保证轨迹至少两帧，回放时不会越界
  if (n < 2) pushFrame();

  const topIndex = topFaceIndex(die, [qx, qy, qz, qw]);
  const topDot = rotateVecY(die.faces[topIndex].normal, [qx, qy, qz, qw]);

  return {
    pos: [px, py, pz],
    quat: [qx, qy, qz, qw],
    vel: [vx, vy, vz],
    omega: [wx, wy, wz],
    tracks: { n, dt, pos: trackPos, quat: trackQuat },
    impacts,
    duration: t,
    asleep,
    nudges,
    topIndex,
    value: die.faces[topIndex].value,
    topDot,
    maxSpeed,
    maxHeight
  };
}

/* ----------------------------------------------------------------------------
 * 造一次投掷的初始状态：从当前停稳的姿态「弹」出去
 *
 * cur: { pos:[x,y,z], quat:[x,y,z,w] }   —— 骰子此刻的姿态（必须是停稳姿态）
 * power: 力度系数（视口越小越轻，别让骰子飞出画面）
 * rnd: [0,1) 随机源，测试时可注入固定种子
 * opts: {
 *   target: [x, z]   期望落点（给了就朝这个点扔，方向由它决定，±17° 抖动）
 *   hSpeed: number   直接指定水平出手速度（定点投掷时由 planThrow 迭代出来）
 * }
 *
 * 没有 target 时：大致朝桌面中心扔、方向左右偏 ±50°，
 * 连续投掷会围绕中心做有界随机游走，不会越掷越偏。
 * -------------------------------------------------------------------------- */
export function createThrow(cur, power = 1, rnd = Math.random, opts = {}){
  // ---- 出手方向 ----
  let ang;
  if (opts.target){
    const tx = opts.target[0] - cur.pos[0];
    const tz = opts.target[1] - cur.pos[2];
    // 有落点时方向要准，只给 ±8° 的抖（横向误差会直接被落点误差放大）
    ang = Math.atan2(tz, tx) + (rnd() - 0.5) * 0.28;
  } else {
    const dx = -cur.pos[0], dz = -cur.pos[2];
    const distC = Math.hypot(dx, dz);
    if (distC > 0.18){
      const spread = 1.75 * Math.max(0.12, Math.min(1, 1 - (distC - 0.45) / 0.95));
      ang = Math.atan2(dz, dx) + (rnd() - 0.5) * spread;
    } else {
      ang = rnd() * Math.PI * 2;
    }
  }
  const ux = Math.cos(ang), uz = Math.sin(ang);

  const hSpeed = opts.hSpeed != null ? opts.hSpeed : (0.24 + rnd() * 0.44) * power;
  const lift = (5.15 + rnd() * 2.00) * (0.82 + 0.18 * power);

  // 自旋轴 = 与出手方向垂直的水平轴 → 向前翻跟头，骰子顺着扔出去的方向滚
  const spin = (8.4 + rnd() * 7.8) * (0.82 + 0.18 * power);
  const tilt = (rnd() - 0.5) * 0.8;                 // 轴略微歪一点，避免每次一个样
  const ca = Math.cos(tilt), sa = Math.sin(tilt);
  const ax = uz * ca + ux * sa;
  const az = uz * sa - ux * ca;

  return {
    pos: [cur.pos[0], cur.pos[1], cur.pos[2]],
    quat: [cur.quat[0], cur.quat[1], cur.quat[2], cur.quat[3]],
    vel: [ux * hSpeed, lift, uz * hSpeed],
    // 竖直分量给一点点就够：给多了骰子会落定后「单脚打转」
    omega: [ax * spin, (rnd() - 0.5) * 1.1, az * spin]
  };
}

/* ----------------------------------------------------------------------------
 * 定点投掷：先在安全圈里挑一个落点 → 反推水平初速 → 预演 → 校准
 *
 * 为什么落点要选「绝对坐标」而不是「往某个方向扔多远」：
 * 只有在圈内指定落点，连续投掷才不会一步步往外漂（实测按方向扔会稳定漂到圈外）。
 *
 * 为什么要闭环校准：落地后还会蹦几下、滚一段，位移和出手速度不是简单比例，
 * 所以按「实际走了多少 / 期望走多少」迭代修正，一般 2~3 次收敛。
 *
 * 返回与 simulateThrow 相同的结构。
 * -------------------------------------------------------------------------- */
export function planThrow(die, cur, opt = {}){
  const power = opt.power != null ? opt.power : 1;
  const rnd = opt.rnd || Math.random;
  const safe = opt.safeRadius != null ? opt.safeRadius : 1.15;
  const simOpt = opt.sim || {};
  const attempts = opt.attempts || 6;
  const tol = opt.tolerance || 0.35;

  const px = cur.pos[0], pz = cur.pos[2];
  const pc = Math.hypot(px, pz);

  // 落点方向：从骰子出发、穿过桌面中心（骰子就在中心时随便挑）
  let ux, uz;
  if (pc > 0.25){ ux = -px / pc; uz = -pz / pc; }
  else { const a = rnd() * Math.PI * 2; ux = Math.cos(a); uz = Math.sin(a); }

  // 方向抖动 ±26°，免得每次都从正中穿过
  {
    const j = (rnd() - 0.5) * 0.90;
    const c = Math.cos(j), s = Math.sin(j);
    const nx = ux * c - uz * s;
    const nz = ux * s + uz * c;
    ux = nx; uz = nz;
  }

  // 落点：安全圈内，半径 0.55~0.95 倍（越靠外，这一掷走得越远）
  const tr = safe * (0.55 + rnd() * 0.40);
  const tx = ux * tr, tz = uz * tr;

  const want = Math.hypot(tx - px, tz - pz);   // 需要走多远
  let hs = Math.max(0.2, want * 0.45);
  let best = null, bestErr = Infinity;
  for (let i = 0; i < attempts; i++){
    const init = createThrow(cur, power, rnd, { target: [tx, tz], hSpeed: hs });
    const s = simulateThrow(die, init, simOpt);
    const moved = Math.hypot(s.pos[0] - px, s.pos[2] - pz);
    const err = Math.hypot(s.pos[0] - tx, s.pos[2] - tz);   // 离落点还差多少
    if (err < bestErr){ bestErr = err; best = s; }
    if (err < tol) return s;
    // 按比例修正：走少了就加速
    const k = moved > 1e-4 ? Math.min(2.0, Math.max(0.5, want / moved)) : 2.0;
    hs = Math.max(0.15, Math.min(4.5, hs * k));
  }
  return best;
}

/* ---------------------------------------------------------------- 小工具 */
function sub(a, b){ return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b){
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot(a, b){ return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function normalize(a){
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}





