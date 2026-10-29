// ============================================================
// SMARTBLOOD WEB APP
// Firebase Authentication + Firestore + QR Donor Profiles
// ============================================================

import { initializeApp } from
    "https://www.gstatic.com/firebasejs/12.19.0/firebase-app.js";

import {
    getAuth,
    GoogleAuthProvider,
    signInWithPopup,
    signInWithPhoneNumber,
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    sendPasswordResetEmail,
    onAuthStateChanged,
    signOut
} from
    "https://www.gstatic.com/firebasejs/12.19.0/firebase-auth.js";

import {
    getFirestore,
    collection,
    doc,
    setDoc,
    addDoc,
    getDocs,
    getDoc,
    updateDoc,
    query,
    where,
    serverTimestamp
} from
    "https://www.gstatic.com/firebasejs/12.19.0/firebase-firestore.js";
// ============================================================
// DONOR AVAILABILITY
// ============================================================

async function getMyDonorRecord() {

    const user = auth.currentUser;

    if (!user) {
        return null;
    }

    const q = query(
        collection(db, "donors"),
        where("ownerUid", "==", user.uid)
    );

    const snapshot = await getDocs(q);

    if (snapshot.empty) {
        return null;
    }

    const donorDocument = snapshot.docs[0];

    return {
        id: donorDocument.id,
        ...donorDocument.data()
    };
}


async function donorStatus() {

    showLoading("Loading your donor status...");

    try {

        const donor = await getMyDonorRecord();

        if (!donor) {

            app.innerHTML = `
                <section class="page">

                    ${pageTop(
                        "My donor status",
                        "Register as a donor to manage your availability."
                    )}

                    <div class="panel">

                        <h2>You are not registered as a donor yet.</h2>

                        <p class="page-subtitle">
                            Register as a donor to appear in emergency
                            blood searches.
                        </p>

                        <button
                            class="primary"
                            data-route="register"
                        >
                            Register as a donor
                        </button>

                    </div>

                </section>
            `;

            bindCommon();
            return;
        }


        const isAvailable =
            donor.availability !== "unavailable";


        app.innerHTML = `

            <section class="page">

                ${pageTop(
                    "My donor status",
                    "Control whether you are currently available to help."
                )}


                <div class="form-layout">

                    <div class="panel">

                        <div
                            style="
                                display:flex;
                                align-items:center;
                                gap:18px;
                                margin-bottom:24px;
                            "
                        >

                            <div
                                class="blood-badge"
                                style="
                                    width:64px;
                                    height:64px;
                                    font-size:18px;
                                "
                            >
                                ${esc(donor.bloodGroup)}
                            </div>

                            <div>

                                <h2 style="margin:0;">
                                    ${esc(donor.name)}
                                </h2>

                                <p style="margin:6px 0 0;">
                                    ${esc(donor.location)}
                                </p>

                            </div>

                        </div>


                        <div
                            style="
                                padding:24px;
                                border-radius:18px;
                                background:${
                                    isAvailable
                                        ? "#ecfdf3"
                                        : "#f5f5f5"
                                };
                                border:1px solid ${
                                    isAvailable
                                        ? "#bbf7d0"
                                        : "#e5e5e5"
                                };
                                margin-bottom:24px;
                            "
                        >

                            <div
                                style="
                                    font-size:20px;
                                    font-weight:700;
                                    margin-bottom:8px;
                                "
                            >

                                ${
                                    isAvailable
                                        ? "🟢 You are Available"
                                        : "⚪ You are Unavailable"
                                }

                            </div>


                            <p style="margin:0;">

                                ${
                                    isAvailable
                                        ? "You can receive emergency blood requests."
                                        : "You will not be considered for new emergency requests."
                                }

                            </p>

                        </div>


                        <button
                            class="${
                                isAvailable
                                    ? "secondary"
                                    : "primary"
                            }"
                            id="availability-toggle"
                        >

                            ${
                                isAvailable
                                    ? "Mark as Unavailable"
                                    : "Mark as Available"
                            }

                        </button>

                    </div>


                    <aside class="side-note">

                        <h3>How this works</h3>

                        <p>
                            When you are marked as available,
                            SmartBlood can include you when a patient
                            searches for compatible donors.
                        </p>

                        <ul>

                            <li>
                                You can change your status anytime.
                            </li>

                            <li>
                                Unavailable donors won't receive new
                                emergency requests.
                            </li>

                            <li>
                                Your availability does not determine
                                medical eligibility.
                            </li>

                        </ul>

                    </aside>

                </div>

            </section>
        `;


        const toggleButton =
            document.querySelector("#availability-toggle");


        toggleButton.onclick =
            async () => {

                toggleButton.disabled = true;

                toggleButton.textContent =
                    "Updating...";


                try {

                    const newAvailability =
                        isAvailable
                            ? "unavailable"
                            : "available";


                    await updateDoc(
                        doc(
                            db,
                            "donors",
                            donor.id
                        ),
                        {
                            availability:
                                newAvailability
                        }
                    );


                    toast(
                        newAvailability === "available"
                            ? "You are now available."
                            : "You are now unavailable."
                    );


                    await donorStatus();


                } catch (error) {

                    console.error(
                        "AVAILABILITY UPDATE ERROR:",
                        error
                    );

                    toast(
                        "Could not update availability."
                    );

                    toggleButton.disabled = false;

                    toggleButton.textContent =
                        isAvailable
                            ? "Mark as Unavailable"
                            : "Mark as Available";
                }

            };


        bindCommon();


    } catch (error) {

        console.error(
            "DONOR STATUS ERROR:",
            error
        );

        app.innerHTML = `

            <section class="page">

                <h1 class="page-title">
                    Could not load donor status
                </h1>

                <p class="page-subtitle">
                    ${esc(error.message)}
                </p>

                <button
                    class="primary"
                    data-route="home"
                >
                    Back home
                </button>

            </section>
        `;

        bindCommon();

    }
}


const firebaseConfig = {
    apiKey: "AIzaSyCjE4ZuhMNheFI9sxPZp95XIdlffPHy6vc",
    authDomain: "smart-blood-40f3d.firebaseapp.com",
    projectId: "smart-blood-40f3d",
    storageBucket: "smart-blood-40f3d.firebasestorage.app",
    messagingSenderId: "834283253666",
    appId: "1:834283253666:web:3704d889261e72422b0cfb",
    measurementId: "G-XD2X1VQTDE"
};

const firebaseApp = initializeApp(firebaseConfig);

const auth = getAuth(firebaseApp);
const db = getFirestore(firebaseApp);


const groups = [
    "A+",
    "A-",
    "B+",
    "B-",
    "AB+",
    "AB-",
    "O+",
    "O-"
];


// ============================================================
// BASIC ELEMENTS
// ============================================================

const app = document.querySelector("#app");
const modalRoot = document.querySelector("#modal-root");



// ============================================================
// HELPERS
// ============================================================

const esc = value =>
    String(value ?? "").replace(
        /[&<>'"]/g,
        char => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            "'": "&#39;",
            '"': "&quot;"
        }[char])
    );


const groupOptions = (label = "Choose blood group") => {

    return `
        <option value="">${label}</option>

        ${groups
            .map(group =>
                `<option value="${group}">${group}</option>`
            )
            .join("")}
    `;
};


function route(name) {

    location.hash =
        name === "home"
            ? ""
            : `#${name}`;
}


function generateToken() {

    if (crypto.randomUUID) {
        return crypto.randomUUID();
    }

    return (
        Date.now().toString(36) +
        Math.random().toString(36).substring(2)
    );
}


function showLoading(message = "Loading...") {

    app.innerHTML = `
        <section class="page">
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <p>${esc(message)}</p>
            </div>
        </section>
    `;
}


// ============================================================
// AUTHENTICATION PAGES
// ============================================================

function loginPage() {
    return `
        <div class="login-brand">
            <div class="brand" aria-label="e-SmartBlood">
                <span class="brand-mark">+</span>
                <span class="brand-name">
                    <span style="color: #ba1a1a;">e</span>-Smart<span>Blood</span>
                </span>
            </div>
        </div>

        <section class="page auth-page">
            <div class="auth-card">

                <p class="eyebrow">
                    <span style="color: #ba1a1a;">e</span>-SMARTBLOOD
                </p>

                <h1>Welcome back</h1>

                <p class="page-subtitle">
                    Sign in to access the e-SmartBlood donor network.
                </p>

                <!-- EMAIL LOGIN -->
                <div id="email-login-panel" class="login-panel active">
                    <form id="login-form">

                        <div class="field">
                            <label for="login-email">Email</label>
                            <input
                                id="login-email"
                                type="email"
                                required
                                autocomplete="email"
                                placeholder="Enter your email"
                            />
                        </div>

                        <div class="field">
                            <label for="login-password">Password</label>

                            <div class="password-input-wrapper">
                                <input
                                    id="login-password"
                                    type="password"
                                    required
                                    autocomplete="current-password"
                                    placeholder="Enter your password"
                                />

                                <button
                                    type="button"
                                    class="password-toggle"
                                    data-password-toggle="login-password"
                                    aria-label="Show password"
                                >
                                    👁
                                </button>
                            </div>
                        </div>

                        <div class="forgot-password-row">
                            <button
                                type="button"
                                class="text-button"
                                id="forgot-password"
                            >
                                Forgot password?
                            </button>
                        </div>

                        <button class="primary" type="submit">
                            Sign in
                        </button>

                    </form>

                    <p id="login-error" class="error-message"></p>
                </div>

                <!-- GOOGLE LOGIN -->
                <div class="auth-divider">
                    <span>OR</span>
                </div>

                <button
                    class="google-btn"
                    id="google-login"
                    type="button"
                >
                    Continue with Google
                </button>

                <!-- CREATE ACCOUNT -->
                <p class="auth-switch">
                    Don't have an account?

                    <button
                        class="text-button"
                        data-route="signup"
                        type="button"
                    >
                        Create account
                    </button>
                </p>

            </div>
        </section>
    `;
}
function signupPage() {

    return `
        <section class="page auth-page">

            <div class="auth-card">

                <p class="eyebrow">
                    e-SMARTBLOOD
                </p>

                <h1>
                    Create account
                </h1>

                <p class="page-subtitle">
                    Create your account to access the donor network.
                </p>


                <form id="signup-form">

                    <div class="field">

                        <label for="signup-email">
                            Email
                        </label>

                        <input
                            id="signup-email"
                            type="email"
                            required
                            autocomplete="email"
                            placeholder="Enter your email"
                        />

                    </div>


                    <div class="field">

    <label for="signup-password">
        Password
    </label>

    <div class="password-input-wrapper">

        <input
            id="signup-password"
            type="password"
            minlength="8"
            required
            autocomplete="new-password"
            placeholder="Create a strong password"
        />

        <button
            type="button"
            class="password-toggle"
            data-password-toggle="signup-password"
            aria-label="Show password"
        >
            👁
        </button>

    </div>

</div>


                    <div class="password-requirements">

                        <p class="password-title">
                            Password must contain:
                        </p>

                        <div
                            id="password-length"
                            class="password-rule"
                        >
                            <span class="rule-icon">○</span>
                            At least 8 characters
                        </div>

                        <div
                            id="password-uppercase"
                            class="password-rule"
                        >
                            <span class="rule-icon">○</span>
                            One uppercase letter (A-Z)
                        </div>

                        <div
                            id="password-lowercase"
                            class="password-rule"
                        >
                            <span class="rule-icon">○</span>
                            One lowercase letter (a-z)
                        </div>

                        <div
                            id="password-number"
                            class="password-rule"
                        >
                            <span class="rule-icon">○</span>
                            One number (0-9)
                        </div>

                        <div
                            id="password-special"
                            class="password-rule"
                        >
                            <span class="rule-icon">○</span>
                            One special character (!@#$%)
                        </div>

                    </div>


                    <button
                        class="primary"
                        type="submit"
                        id="signup-submit"
                    >
                        Create account
                    </button>

                </form>


                <p
                    id="signup-error"
                    class="error-message"
                ></p>


                <p class="auth-switch">

                    Already have an account?

                    <button
                        class="text-button"
                        data-route="login"
                    >
                        Sign in
                    </button>

                </p>

            </div>

        </section>
    `;
}
// ============================================================
// PHONE OTP LOGIN
// ============================================================










// ============================================================
// AUTH FUNCTIONS
// ============================================================
function getAuthError(error) {
    const messages = {
        "auth/invalid-email":
            "Please enter a valid email address.",

        "auth/invalid-credential":
            "Incorrect email or password. Please check your details.",

        "auth/user-not-found":
            "No account exists with this email.",

        "auth/wrong-password":
            "Incorrect password. Please try again.",

        "auth/too-many-requests":
            "Too many login attempts. Please wait before trying again.",

        "auth/network-request-failed":
            "Network error. Please check your internet connection.",

        "auth/user-disabled":
            "This account has been disabled."
    };

    return messages[error.code] ||
        "Login failed. Please try again.";
}

async function loginWithGoogle() {
    const button = document.querySelector("#google-login");

    try {
        if (button) {
            button.disabled = true;
            button.textContent = "Connecting to Google...";
        }

        const provider = new GoogleAuthProvider();

        provider.setCustomParameters({
            prompt: "select_account"
        });

        const result = await signInWithPopup(auth, provider);

        console.log("GOOGLE LOGIN SUCCESS:", result.user.email);

        toast("Google login successful.");

        location.hash = "";
        await render();

    } catch (error) {
        console.error(
            "GOOGLE LOGIN ERROR:",
            error.code,
            error.message
        );

        const messages = {
            "auth/unauthorized-domain":
                "This website domain is not authorized in Firebase.",

            "auth/operation-not-allowed":
                "Google Sign-In is not enabled in Firebase Authentication.",

            "auth/popup-blocked":
                "Your browser blocked the Google login popup. Allow popups and try again.",

            "auth/popup-closed-by-user":
                "The Google login window was closed before sign-in finished.",

            "auth/network-request-failed":
                "Network error. Check your internet connection.",

            "auth/cancelled-popup-request":
                "A Google login popup is already open. Close it and try again."
        };

        const message =
            messages[error.code] ||
            `Google login failed: ${error.code || error.message}`;

        toast(message);

    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = "Continue with Google";
        }
    }
}
async function loginUser(email, password) {
    const errorElement = document.querySelector("#login-error");
    const loginButton = document.querySelector(
        '#login-form button[type="submit"]'
    );

    try {
        if (errorElement) {
            errorElement.textContent = "";
        }

        if (loginButton) {
            loginButton.disabled = true;
            loginButton.textContent = "Signing in...";
        }

        const userCredential = await signInWithEmailAndPassword(
            auth,
            email.trim(),
            password
        );

        console.log(
            "LOGIN SUCCESS:",
            userCredential.user.email
        );

        toast("Login successful.");

        location.hash = "";
        await render();

    } catch (error) {
        console.error("LOGIN ERROR:", error.code, error.message);

        const messages = {
            "auth/invalid-email": "Please enter a valid email address.",
            "auth/user-not-found": "No account exists with this email.",
            "auth/wrong-password": "Incorrect email or password.",
            "auth/invalid-credential": "Incorrect email or password.",
            "auth/too-many-requests": "Too many attempts. Please try again later.",
            "auth/network-request-failed": "Network error. Check your internet connection.",
            "auth/operation-not-allowed": "Email and password sign-in is not enabled in Firebase."
        };

        const message =
            messages[error.code] ||
            "Login failed. Please try again.";

        if (errorElement) {
            errorElement.textContent = message;
        } else {
            toast(message, "error");
        }

    } finally {
        if (loginButton) {
            loginButton.disabled = false;
            loginButton.textContent = "Sign in";
        }
    }
}

// ============================================================
// HOME
// ============================================================

async function getDonorCount() {

    try {

        const snapshot =
            await getDocs(
                collection(db, "donors")
            );

        return snapshot.size;

    } catch (error) {

        console.error(error);

        return 0;

    }
}


async function home() {

    const count =
        await getDonorCount();

    return `
        <section class="page">

            <div class="hero">

                <div>

                    <p class="eyebrow">
                        A local blood donor network
                    </p>

                    <h1>
                        Every donor can be
                        <em>someone's hope.</em>
                    </h1>

                    <p class="hero-copy">
                        SmartBlood makes it simple to register,
                        discover compatible donors, and act quickly
                        in an emergency.
                    </p>


                    <div class="button-row">

                        <button
                            class="primary"
                            data-route="emergency"
                        >
                            Request blood now
                        </button>

                        <button
                            class="secondary"
                            data-route="register"
                        >
                            Become a donor
                        </button>

                    </div>

                </div>


                <aside class="hero-card">

                    <div class="pulse">
                        ♥
                    </div>

                    <h2>
                        Help can be closer than you think.
                    </h2>

                    <p>
                        Connect with voluntary donors
                        in your community.
                    </p>


                    <div class="hero-stat">

                        <div>
                            <b>${count}</b>
                            <span>listed donors</span>
                        </div>

                        <div>
                            <b>8</b>
                            <span>blood groups</span>
                        </div>

                    </div>

                </aside>

            </div>


            <div class="section-heading">

                <div>

                    <p class="eyebrow">
                        How can we help?
                    </p>

                    <h2>
                        Make the right move, faster.
                    </h2>

                </div>

            </div>


            <div class="quick-grid">

                <button
                    class="quick-card"
                    data-route="emergency"
                >

                    <div class="quick-icon">
                        ⚡
                    </div>

                    <h3>
                        Emergency request
                    </h3>

                    <p>
                        Find matching donors for
                        an urgent need.
                    </p>

                </button>


                <button
                    class="quick-card"
                    data-route="find"
                >

                    <div class="quick-icon">
                        ⌕
                    </div>

                    <h3>
                        Find donors
                    </h3>

                    <p>
                        Search listed donors
                        by blood group.
                    </p>

                </button>


                <button
                    class="quick-card"
                    data-route="register"
                >

                    <div class="quick-icon">
                        ♡
                    </div>

                    <h3>
                        Register as donor
                    </h3>

                    <p>
                        Join the network and
                        help when you can.
                    </p>

                </button>


                <button
                    class="quick-card"
                    data-route="availability"
                >

                    <div class="quick-icon">
                        ▦
                    </div>

                    <h3>
                        Blood availability
                    </h3>

                    <p>
                        See the community donor
                        count by group.
                    </p>

                </button>

            </div>

        </section>
    `;
}


// ============================================================
// COMMON PAGE HEADER
// ============================================================

function pageTop(title, subtitle) {

    return `
        <div class="page-top">

            <button
                class="back back-button"
                type="button"
                data-route="home"
                aria-label="Back to home"
            >
                ←
            </button>

            <div>

                <h1 class="page-title">
                    ${esc(title)}
                </h1>

                <p class="page-subtitle">
                    ${esc(subtitle)}
                </p>

            </div>

        </div>
    `;
    
}


// ============================================================
// DONOR CARD
// ============================================================

function donorCard(d) {

    return `
        <article class="donor-card">

            <div class="donor-main">

                <div class="blood-badge">
                    ${esc(d.bloodGroup)}
                </div>

                <div>

                    <h3>
                        ${esc(d.name)}
                    </h3>

                    <p>
                        ${esc(d.location)}
                        ·
                        ${esc(d.phone || "Contact through SmartBlood")}
                    </p>

                    ${
                        typeof d.distanceKm === "number"
                            ? `
                                <p style="
                                    margin-top:6px;
                                    font-weight:700;
                                    color:#b51623;
                                ">
                                    📍
                                    ${d.distanceKm < 1
                                        ? `${Math.round(d.distanceKm * 1000)} m away`
                                        : `${d.distanceKm.toFixed(1)} km away`
                                    }
                                </p>
                            `
                            : ""
                    }

                    <small style="
                        display:inline-block;
                        margin-top:6px;
                        font-weight:700;
                        color:${d.availability === "unavailable" ? "#777" : "#16803c"};
                    ">
                        ${
                            d.availability === "unavailable"
                                ? "Unavailable"
                                : "Available"
                        }
                    </small>

                </div>

            </div>


            <button
                class="outline"
                data-profile="${esc(d.id)}"
            >
                View donor profile
            </button>

        </article>
    `;
}

// ============================================================
// REGISTER DONOR
// ============================================================

function register() {

    return `
        <section class="page">

            ${pageTop(
                "Register as a donor",
                "Your details will help people find a compatible donor in an emergency."
            )}

            <div class="form-layout">

                <form
                    class="panel"
                    id="register-form"
                >

                    <div class="field-grid">

                        <div class="field">

                            <label for="name">
                                Full name
                            </label>

                            <input
                                id="name"
                                required
                                placeholder="Your full name"
                            />

                        </div>


                        <div class="field">

                            <label for="age">
                                Age
                            </label>

                            <input
                                id="age"
                                type="number"
                                required
                                min="18"
                                max="65"
                                placeholder="Your age"
                            />

                        </div>

                    </div>


                    <div class="field-grid">

                        <div class="field">

                            <label for="phone">
                                Phone number
                            </label>

                            <input
                                id="phone"
                                required
                                inputmode="tel"
                                maxlength="10"
                                placeholder="10-digit mobile number"
                            />

                        </div>


                        <div class="field">

                            <label for="blood">
                                Blood group
                            </label>

                            <select
                                id="blood"
                                required
                            >
                                ${groupOptions()}
                            </select>

                        </div>

                    </div>


                    <div class="field">

                        <label for="location">
                            Location
                        </label>

                        <input
                            id="location"
                            required
                            placeholder="Area, city"
                        />

                    </div>


                    <div class="field">

                        <label for="medical-history">
                            Medical history / health conditions
                        </label>

                        <textarea
                            id="medical-history"
                            rows="4"
                            placeholder="Optional: Mention relevant health conditions or medical history"
                        ></textarea>

                    </div>


                    <button
                        class="primary"
                        type="submit"
                    >
                        Continue to consent
                    </button>

                </form>


                <aside class="side-note">

                    <h3>
                        Your privacy matters
                    </h3>

                    <p>
                        Your contact details are used to help
                        people locate a compatible donor.
                    </p>

                    <ul>

                        <li>
                            Join only if you are willing
                            to be contacted.
                        </li>

                        <li>
                            Keep your availability up to date.
                        </li>

                        <li>
                            SmartBlood does not decide
                            medical eligibility.
                        </li>

                    </ul>

                </aside>

            </div>

        </section>
    `;
}


// ============================================================
// FIND DONORS
// ============================================================

function find() {

    return `
        <section class="page">

            ${pageTop(
                "Find blood donors",
                "Search the volunteer directory by the required blood group."
            )}


            <div class="panel">

                <div class="search-bar">

                    <div class="field">

                        <label for="search-group">
                            Required blood group
                        </label>

                        <select id="search-group">
                            ${groupOptions("All blood groups")}
                        </select>

                    </div>


                    <button
                        class="primary"
                        id="search-btn"
                    >
                        Search donors
                    </button>

                </div>


                <div
                    id="search-results"
                    class="donor-list"
                >

                    <p class="empty">
                        Choose a blood group to view
                        matching donor profiles.
                    </p>

                </div>

            </div>

        </section>
    `;
}
function calculateDistance(
    latitude1,
    longitude1,
    latitude2,
    longitude2
) {
    const earthRadius = 6371;

    const lat1 = latitude1 * Math.PI / 180;
    const lat2 = latitude2 * Math.PI / 180;

    const deltaLat =
        (latitude2 - latitude1) * Math.PI / 180;

    const deltaLon =
        (longitude2 - longitude1) * Math.PI / 180;

    const a =
        Math.sin(deltaLat / 2) *
        Math.sin(deltaLat / 2) +
        Math.cos(lat1) *
        Math.cos(lat2) *
        Math.sin(deltaLon / 2) *
        Math.sin(deltaLon / 2);

    const c =
        2 *
        Math.atan2(
            Math.sqrt(a),
            Math.sqrt(1 - a)
        );

    return earthRadius * c;
}


// ============================================================
// SEARCH FIRESTORE DONORS
// ============================================================

async function searchDonors(bloodGroup = "") {

    try {

        console.log("SEARCH DONORS STARTED");
        console.log("Blood group:", bloodGroup);
        console.log("Current user:", auth.currentUser?.email);

        const snapshot = await getDocs(
            collection(db, "donors")
        );

        console.log(
            "Total donor documents:",
            snapshot.size
        );

        let donors = snapshot.docs.map(document => ({
            id: document.id,
            ...document.data()
        }));


        // Only show active and available donors
        donors = donors.filter(
            donor =>
                donor.active === true &&
                donor.availability !== "unavailable"
        );


        // Filter blood group in JavaScript
        // instead of using a Firestore composite query
        if (bloodGroup) {

            donors = donors.filter(
                donor =>
                    donor.bloodGroup === bloodGroup
            );

        }


        console.log(
            "Matching donors:",
            donors.length
        );


        // ==========================================
        // GET USER LOCATION
        // ==========================================

        let patientLatitude = null;
        let patientLongitude = null;


        try {

            if (
                navigator.geolocation
            ) {

                const position =
                    await new Promise(
                        (resolve, reject) => {

                            navigator.geolocation.getCurrentPosition(
                                resolve,
                                reject,
                                {
                                    enableHighAccuracy: true,
                                    timeout: 10000,
                                    maximumAge: 0
                                }
                            );

                        }
                    );


                patientLatitude =
                    position.coords.latitude;

                patientLongitude =
                    position.coords.longitude;


                console.log(
                    "📍 Patient location:",
                    patientLatitude,
                    patientLongitude
                );

            }

        } catch (locationError) {

            console.warn(
                "Location unavailable. Showing donors without distance.",
                locationError
            );

        }


        // ==========================================
        // CALCULATE DISTANCE
        // ==========================================

        const donorsWithDistance =
            donors.map(donor => {

                if (
                    patientLatitude === null ||
                    patientLongitude === null ||
                    typeof donor.latitude !== "number" ||
                    typeof donor.longitude !== "number"
                ) {

                    return {
                        ...donor,
                        distanceKm: null
                    };

                }


                const distance =
                    calculateDistance(
                        patientLatitude,
                        patientLongitude,
                        donor.latitude,
                        donor.longitude
                    );


                return {
                    ...donor,
                    distanceKm: distance
                };

            });


        // ==========================================
        // NEAREST DONOR FIRST
        // ==========================================

        donorsWithDistance.sort(
            (a, b) => {

                if (
                    a.distanceKm === null
                ) {
                    return 1;
                }

                if (
                    b.distanceKm === null
                ) {
                    return -1;
                }

                return (
                    a.distanceKm -
                    b.distanceKm
                );

            }
        );


        console.log(
            "FINAL DONOR RESULTS:",
            donorsWithDistance
        );


        return donorsWithDistance;

    } catch (error) {

        console.error(
            "SEARCH DONORS FIREBASE ERROR:",
            error
        );

        throw error;

    }

}

// ============================================================
// AVAILABILITY
// ============================================================

async function getAvailabilityCounts() {

    const counts = {};

    groups.forEach(group => {

        counts[group] = 0;

    });


    try {

        const snapshot =
            await getDocs(
                query(
                    collection(db, "donors"),
                    where("active", "==", true)
                )
            );


        snapshot.forEach(document => {

    const donor = document.data();

    if (
        counts[donor.bloodGroup] !== undefined &&
        donor.availability !== "unavailable"
    ) {

        counts[donor.bloodGroup]++;

    }

});


    } catch (error) {

        console.error(error);

    }


    return counts;
}


async function availability() {

    const counts =
        await getAvailabilityCounts();


    const list =
        groups
            .map((group, index) => {

                const count =
                    counts[group] || 0;


                return `
                    <div
                        class="availability-card"
                        style="animation-delay:${index * 35}ms"
                    >

                        <b>
                            ${group}
                        </b>

                        <span>
                            ${
                                count
                                    ? `${count} donor${count > 1 ? "s" : ""} available`
                                    : "No donors listed yet"
                            }
                        </span>

                    </div>
                `;

            })
            .join("");


    return `
        <section class="page">

            ${pageTop(
                "Blood availability",
                "A live view of voluntary donors in the SmartBlood directory."
            )}

            <div class="availability">
                ${list}
            </div>

        </section>
    `;
}


// ============================================================
// EMERGENCY REQUEST PAGE
// ============================================================

function emergency() {

    return `
        <section class="page">

            ${pageTop(
                "Emergency blood request",
                "Submit the request and we'll surface compatible listed donors."
            )}

            <div class="form-layout">

                <form
                    class="panel"
                    id="emergency-form"
                >

                    <div class="field-grid">

                        <div class="field">

                            <label for="patient">
                                Patient name
                            </label>

                            <input
                                id="patient"
                                required
                                placeholder="Patient full name"
                            />

                        </div>


                        <div class="field">

                            <label for="patient-age">
                                Patient age
                            </label>

                            <input
                                id="patient-age"
                                type="number"
                                required
                                min="0"
                                max="120"
                                placeholder="Patient age"
                            />

                        </div>


                        <div class="field">

                            <label for="required-group">
                                Required blood group
                            </label>

                            <select
                                id="required-group"
                                required
                            >
                                ${groupOptions()}
                            </select>

                        </div>

                    </div>


                    <div class="field-grid">

                        <div class="field">

                            <label for="hospital">
                                Hospital name
                            </label>

                            <input
                                id="hospital"
                                required
                                placeholder="Hospital name"
                            />

                        </div>


                        <div class="field">

                            <label for="request-location">
                                Location
                            </label>

                            <input
                                id="request-location"
                                required
                                placeholder="Area, city"
                            />

                        </div>

                    </div>


                    <div class="field">

                        <label for="patient-medical-history">
                            Medical history / health conditions
                        </label>

                        <textarea
                            id="patient-medical-history"
                            rows="4"
                            placeholder="Optional: Mention relevant medical conditions"
                        ></textarea>

                    </div>


                    <button
                        class="primary"
                        type="submit"
                    >
                        Find matching donors
                    </button>

                </form>


                <aside class="side-note">

                    <h3>
                        For urgent situations
                    </h3>

                    <p>
                        Call local emergency services and the
                        hospital's blood bank immediately.
                        SmartBlood is a directory, not an
                        emergency or medical service.
                    </p>

                </aside>

            </div>


            <div
                id="emergency-results"
                class="donor-list"
            ></div>

        </section>
    `;
}


// ============================================================
// SAVE DONOR TO FIRESTORE
// ============================================================

async function saveDonor(data) {
    console.log("1. Starting donor registration...");

    const user = auth.currentUser;

    if (!user) {
        closeModal();
        toast("Please sign in before registering as a donor.");
        route("login");
        return;
    }

    console.log("2. Logged in user:", user.uid);

    let latitude = null;
    let longitude = null;

    // ------------------------------------------------------------
    // GET LOCATION (OPTIONAL)
    // ------------------------------------------------------------

    try {
        if (navigator.geolocation) {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(
                    resolve,
                    reject,
                    {
                        enableHighAccuracy: false,
                        timeout: 10000,
                        maximumAge: 30000
                    }
                );
            });

            latitude = position.coords.latitude;
            longitude = position.coords.longitude;

            console.log("GPS coordinates:", latitude, longitude);
        } else {
            console.warn("Geolocation is not supported by this browser.");
        }
    } catch (locationError) {
        console.warn(
            "GPS unavailable. Continuing with the entered location:",
            locationError.message
        );

        // Continue registration using data.location.
        // Latitude and longitude remain null.
    }

    // ------------------------------------------------------------
    // CREATE DONOR RECORD
    // ------------------------------------------------------------

    const token = generateToken();
    const donorRef = doc(collection(db, "donors"));

    const donor = {
        name: data.name.trim(),
        age: Number(data.age),
        phone: data.phone.trim(),
        bloodGroup: data.bloodGroup,
        location: data.location.trim(),
        medicalHistory: data.medicalHistory || "",
        latitude: latitude,
        longitude: longitude,
        ownerUid: user.uid,
        email: user.email || "",
        active: true,
        availability: "available",
        publicToken: token,
        createdAt: serverTimestamp()
    };

    // ------------------------------------------------------------
    // SAVE DONOR TO FIRESTORE
    // ------------------------------------------------------------

    try {
        console.log("3. Saving donor record...");

        await setDoc(donorRef, donor);

        console.log("4. Donor record saved:", donorRef.id);

        // Save limited information for the public donor profile.
        await setDoc(doc(db, "publicDonorProfiles", token), {
            donorId: donorRef.id,
            name: donor.name,
            bloodGroup: donor.bloodGroup,
            location: donor.location,
            active: true,
            createdAt: serverTimestamp()
        });

        console.log("5. Public donor profile saved.");

        closeModal();

        successModal({
            ...data,
            name: donor.name,
            bloodGroup: donor.bloodGroup,
            id: donorRef.id,
            token: token
        });

    } catch (error) {
        console.error("DONOR REGISTRATION ERROR:", error);

        closeModal();

        toast(
            "Donor registration failed. Please check your connection and try again."
        );
    }
}

    // ------------------------------------------------------------
    // SAVE DONOR AND PUBLIC PROFILE
    // ------------------------------------------------------------

    try {
        // Create a unique QR token.
        const token = generateToken();

        console.log("3. QR token created.");

        // Create a unique donor document reference.
        const donorRef = doc(collection(db, "donors"));

        console.log("4. Donor document created:", donorRef.id);

        const donor = {
            name: data.name,
            age: Number(data.age),
            phone: data.phone,
            bloodGroup: data.bloodGroup,
            location: data.location,
            medicalHistory: data.medicalHistory || "",

            // Use the actual coordinates obtained above.
            latitude: latitude,
            longitude: longitude,

            ownerUid: user.uid,
            email: user.email || "",

            active: true,
            availability: "available",

            publicToken: token,
            createdAt: serverTimestamp()
        };

        console.log("5. Saving donor to Firestore...");

        await setDoc(donorRef, donor);

        console.log("6. Donor saved successfully!");

        // Create the public profile used by the QR link.
        console.log("7. Creating public donor profile...");

        await setDoc(
            doc(db, "publicDonorProfiles", token),
            {
                donorId: donorRef.id,
                name: data.name,
                bloodGroup: data.bloodGroup,
                location: data.location,
                active: true,
                createdAt: serverTimestamp()
            }
        );

        console.log("8. Public donor profile created!");

        const donorForPopup = {
            ...data,
            id: donorRef.id,
            token: token
        };

        console.log("9. Showing success popup...");

        successModal(donorForPopup);

    } catch (error) {
        console.error("FIREBASE REGISTRATION ERROR:", error);

        closeModal();

        toast(
            "Registration failed: " +
            (error.message || "Unknown Firebase error")
        );
    }



// ============================================================
// SUCCESS POPUP + QR CODE
// ============================================================

function successModal(donor) {
    const token = donor.token;

    // Direct website URL for the donor profile.
    // Scanning the QR code opens donor details in the browser.
    const profileUrl =
        `${location.origin}${location.pathname}#donor/${encodeURIComponent(token)}`;

    // Generate the QR code for the donor profile website.
    const qr =
        `https://api.qrserver.com/v1/create-qr-code/?size=300x300&margin=8&data=${encodeURIComponent(profileUrl)}`;

    modalRoot.innerHTML = `
        <div class="modal-overlay">
            <div class="modal success">

                <div class="success-icon">✓</div>

                <p class="eyebrow">
                    Welcome to e-SmartBlood
                </p>

                <h2>You're now listed!</h2>

                <p>
                    Thank you, ${esc(donor.name)}.
                    Your ${esc(donor.bloodGroup)} donor profile
                    has been registered successfully.
                </p>

                <div class="qr-wrap">
                    <div>
                        <h3>Your Donor QR Code</h3>

                        <small>
                            Scan this QR code to view your donor details
                            directly on the SmartBlood website.
                        </small>
                    </div>

                    <img
                        id="donor-qr-image"
                        src="${qr}"
                        alt="QR code for donor profile"
                        style="
                            display: block;
                            width: 220px;
                            height: 220px;
                            object-fit: contain;
                            margin: 16px auto;
                        "
                    />

                    <p
                        id="qr-error-message"
                        style="display:none;color:#ba1a1a;"
                    >
                        The QR code could not load.
                        Please check your internet connection and try again.
                    </p>
                </div>

                <button
                    class="primary"
                    id="view-profile"
                    type="button"
                >
                    View Donor Details
                </button>

                <button
                    class="text-button"
                    id="close-success"
                    type="button"
                >
                    Return Home
                </button>

            </div>
        </div>
    `;

    // Handle QR image loading errors.
    const qrImage = document.querySelector("#donor-qr-image");
    const qrError = document.querySelector("#qr-error-message");

    qrImage.onerror = () => {
        qrImage.style.display = "none";
        qrError.style.display = "block";
        console.error("Donor QR image failed to load.");
    };

    // Open the donor profile directly on the website.
    document.querySelector("#view-profile").onclick = () => {
        closeModal();
        location.hash = `donor/${encodeURIComponent(token)}`;
    };

    // Return to the home page.
    document.querySelector("#close-success").onclick = () => {
        closeModal();
        route("home");
    };
}
// ============================================================
// PUBLIC QR DONOR PROFILE
// ============================================================
async function getPublicDonor(token) {
    if (!token) {
        console.error("QR ERROR: Missing donor token.");
        return null;
    }

    try {
        const donorToken = decodeURIComponent(token).trim();

        console.log("QR TOKEN:", donorToken);

        const profileRef = doc(
            db,
            "publicDonorProfiles",
            donorToken
        );

        const snapshot = await getDoc(profileRef);

        console.log("PROFILE DOCUMENT EXISTS:", snapshot.exists());

        if (!snapshot.exists()) {
            console.error(
                "QR ERROR: No matching document in publicDonorProfiles.",
                donorToken
            );
            return null;
        }

        const donor = snapshot.data();

        console.log("PROFILE DATA:", donor);

        if (donor.active !== true) {
            console.error(
                "QR ERROR: Donor is inactive.",
                donor.active
            );
            return null;
        }

        return {
            id: snapshot.id,
            ...donor
        };

    } catch (error) {
        console.error(
            "QR FIRESTORE ERROR:",
            error.code,
            error.message
        );
        return null;
    }
}


// ============================================================
// PROFILE FOR AUTHENTICATED SEARCH RESULTS
// ============================================================

async function privateProfile(id) {

    showLoading(
        "Loading donor profile..."
    );


    try {

        const donorRef =
            doc(
                db,
                "donors",
                id
            );


        const snapshot =
            await getDoc(donorRef);


        if (!snapshot.exists()) {

            app.innerHTML = `
                <section class="page profile">

                    <p class="eyebrow">
                        SmartBlood profile
                    </p>

                    <h1 class="page-title">
                        Profile unavailable
                    </h1>

                    <p class="page-subtitle">
                        This donor profile cannot be found.
                    </p>

                    <button
                        class="primary"
                        data-route="home"
                    >
                        Back home
                    </button>

                </section>
            `;

            bindCommon();

            return;

        }


        const donor = {

            id:
                snapshot.id,

            ...snapshot.data()

        };


        app.innerHTML = `

            <section class="page profile">

                <p class="profile-label">
                    ✓ SmartBlood directory profile
                </p>


                <h1 class="page-title">
                    Donor profile
                </h1>


                <p class="page-subtitle">
                    Please contact respectfully and confirm
                    availability before making plans.
                </p>


                ${donorCard(donor)}


                ${
                    donor.publicToken
                        ? `
                            <button
                                class="secondary"
                                id="profile-qr"
                                style="margin-top:18px"
                            >
                                View donor QR
                            </button>
                        `
                        : ""
                }


                <button
                    class="secondary"
                    data-route="find"
                    style="margin-top:18px"
                >
                    Find more donors
                </button>

            </section>
        `;


        const qrButton =
            document.querySelector(
                "#profile-qr"
            );


        if (qrButton) {

            qrButton.onclick =
                () => {

                    showDonorQR(
                        donor
                    );

                };

        }


        bindCommon();

    } catch (error) {

        console.error(error);

        toast(
            "Could not load donor profile."
        );

        route("find");

    }
}


// ============================================================
// QR MODAL
// ============================================================

function showDonorQR(donor) {

    if (!donor.publicToken) {

        toast(
            "QR profile is not available."
        );

        return;

    }


    const url =
        `${location.origin}${location.pathname}#donor/${encodeURIComponent(donor.publicToken)}`;


    const qr =
        `https://api.qrserver.com/v1/create-qr-code/?size=300x300&margin=8&data=${encodeURIComponent(url)}`;


    modalRoot.innerHTML = `

        <div class="modal-overlay">

            <div class="modal">

                <p class="eyebrow">
                    Donor QR
                </p>


                <h2>
                    ${esc(donor.name)}
                </h2>


                <div class="qr-wrap">

                    <div>

                        <h3>
                            Share this QR
                        </h3>

                        <small>
                            Scanning this QR opens the
                            donor's public SmartBlood profile.
                        </small>

                    </div>


                    <img
                        src="${qr}"
                        alt="Donor QR code"
                    />

                </div>


                <button
                    class="primary"
                    id="close-qr"
                >
                    Done
                </button>

            </div>

        </div>
    `;


    document.querySelector(
        "#close-qr"
    ).onclick =
        closeModal;
}
async function notificationsPage() {

    const user = auth.currentUser;

    if (!user) {
        route("login");
        return "";
    }


    try {

        const q =
            query(
                collection(db, "notifications"),
                where(
                    "recipientUid",
                    "==",
                    user.uid
                )
            );


        const snapshot =
            await getDocs(q);


        const notifications =
    (
        await Promise.all(

            snapshot.docs.map(
                async document => {

                    const notification = {
                        id: document.id,
                        ...document.data()
                    };


                    // If this notification is linked
                    // to an emergency request,
                    // check the current request status.
                    if (notification.requestId) {

                        const requestSnapshot =
                            await getDoc(
                                doc(
                                    db,
                                    "bloodRequests",
                                    notification.requestId
                                )
                            );


                        // Request no longer exists
                        if (!requestSnapshot.exists()) {
                            return null;
                        }


                        const request =
                            requestSnapshot.data();


                        // Do NOT show notifications
                        // for declined requests.
                        if (
                            request.status ===
                            "declined"
                        ) {
                            return null;
                        }


                        return {
                            ...notification,
                            requestStatus:
                                request.status
                        };

                    }


                    return notification;

                }
            )

        )
    )
        .filter(
            notification =>
                notification !== null
        )
        .sort(
            (a, b) => {

                const timeA =
                    a.createdAt?.toMillis?.() || 0;

                const timeB =
                    b.createdAt?.toMillis?.() || 0;

                return timeB - timeA;

            }
        );


        return `
            <section class="page">

                ${pageTop(
                    "Notifications",
                    "Emergency blood requests matching your blood group."
                )}


                ${
                    notifications.length === 0

                        ? `
                            <div class="panel">

                                <h2>
                                    No notifications
                                </h2>

                                <p class="page-subtitle">
                                    You don't have any emergency
                                    blood request notifications yet.
                                </p>

                            </div>
                        `

                        : `
                            <div class="donor-list">

                                ${
                                    notifications
                                        .map(
                                            notification => `

                                                <article
                                                    class="panel"
                                                >

                                                    <p class="eyebrow">
                                                        ${
                                                            notification.read
                                                                ? "NOTIFICATION"
                                                                : "NEW EMERGENCY REQUEST"
                                                        }
                                                    </p>


                                                    <h2>
                                                        ${esc(
                                                            notification.title ||
                                                            "Emergency blood request"
                                                        )}
                                                    </h2>


                                                    <p>
                                                        ${esc(
                                                            notification.message ||
                                                            ""
                                                        )}
                                                    </p>


                                                    <div class="notice">

                                                        <p>
                                                            <strong>
                                                                Patient:
                                                            </strong>
                                                            ${esc(
                                                                notification.patientName ||
                                                                "Not provided"
                                                            )}
                                                        </p>


                                                        <p>
                                                            <strong>
                                                                Patient age:
                                                            </strong>
                                                            ${
                                                                notification.patientAge ??
                                                                "Not provided"
                                                            }
                                                        </p>


                                                        <p>
                                                            <strong>
                                                                Blood group:
                                                            </strong>
                                                            ${esc(
                                                                notification.bloodGroup ||
                                                                "Not provided"
                                                            )}
                                                        </p>


                                                        <p>
                                                            <strong>
                                                                Hospital:
                                                            </strong>
                                                            ${esc(
                                                                notification.hospital ||
                                                                "Not provided"
                                                            )}
                                                        </p>


                                                        <p>
                                                            <strong>
                                                                Location:
                                                            </strong>
                                                            ${esc(
                                                                notification.location ||
                                                                "Not provided"
                                                            )}
                                                        </p>


                                                        ${
                                                            typeof notification.distanceKm ===
                                                            "number"
                                                                ? `
                                                                    <p>
                                                                        <strong>
                                                                            Distance:
                                                                        </strong>
                                                                        ${
                                                                            notification.distanceKm <
                                                                            1
                                                                                ? `${Math.round(notification.distanceKm * 1000)} m away`
                                                                                : `${notification.distanceKm.toFixed(1)} km away`
                                                                        }
                                                                    </p>
                                                                `
                                                                : ""
                                                        }

                                                    </div>


                                                    <div class="modal-actions">

                                                        <button
                                                            class="primary notification-request-button"
                                                            type="button"
                                                            data-request-id="${esc(
                                                                notification.requestId ||
                                                                ""
                                                            )}"
                                                        >
                                                            View emergency request
                                                        </button>

                                                    </div>

                                                </article>

                                            `
                                        )
                                        .join("")
                                }

                            </div>
                        `
                }

            </section>
        `;
    } catch (error) {

        console.error(
            "Could not load notifications:",
            error
        );


        return `
            <section class="page">

                ${pageTop(
                    "Notifications",
                    "Emergency blood requests matching your blood group."
                )}

                <div class="panel">

                    <h2>
                        Could not load notifications
                    </h2>

                    <p class="page-subtitle">
                        Please try again.
                    </p>

                </div>

            </section>
        `;
    }
}
function updateHeaderVisibility() {

    const header =
        document.getElementById("siteHeader");

    const nav =
        document.getElementById("siteNav");

    if (!header) {
        return;
    }

    const hash =
        location.hash.slice(1);

    if (
        hash === "login" ||
        hash === "signup"
    ) {

        header.style.display = "flex";

        if (nav) {
            nav.style.display = "none";
        }

        return;
    }

    if (auth.currentUser) {

        header.style.display = "flex";

        if (nav) {
            nav.style.display = "flex";
        }

        return;
    }

    header.style.display = "none";
}
// ============================================================
// MY EMERGENCY REQUESTS
// ============================================================

async function myRequests() {

    const user = auth.currentUser;

    if (!user) {
        route("login");
        return;
    }

    try {

        const requestsQuery = query(
            collection(db, "bloodRequests"),
            where("createdBy", "==", user.uid)
        );

        const snapshot =
            await getDocs(requestsQuery);

        const requests =
            snapshot.docs
                .map(document => ({
                    id: document.id,
                    ...document.data()
                }))
                .sort((a, b) => {

                    const timeA =
                        a.timestamp?.toMillis
                            ? a.timestamp.toMillis()
                            : 0;

                    const timeB =
                        b.timestamp?.toMillis
                            ? b.timestamp.toMillis()
                            : 0;

                    return timeB - timeA;
                });


        if (requests.length === 0) {

            app.innerHTML = `
                <section class="page">

                    ${pageTop(
                        "My emergency requests",
                        "View the blood requests you have submitted."
                    )}

                    <div class="panel">

                        <div class="empty-state">

                            <h2>
                                No emergency requests yet
                            </h2>

                            <p>
                                You haven't created an emergency
                                blood request yet.
                            </p>

                            <button
                                class="primary"
                                data-route="emergency"
                            >
                                Create emergency request
                            </button>

                        </div>

                    </div>

                </section>
            `;

            bindCommon();

            return;
        }


        const requestCards =
            requests
                .map(request => {

                    const date =
                        request.timestamp?.toDate
                            ? request.timestamp
                                .toDate()
                                .toLocaleString()
                            : "Recently";


                    return `
                        <div class="panel request-card">

                            <div
                                style="
                                    display:flex;
                                    justify-content:space-between;
                                    align-items:flex-start;
                                    gap:20px;
                                    margin-bottom:20px;
                                "
                            >

                                <div>

                                    <p class="eyebrow">
                                        Emergency request
                                    </p>

                                    <h2>
                                        ${esc(
                                            request.patientName
                                        )}
                                    </h2>

                                </div>


                                <span
                                    style="
                                        display:inline-flex;
                                        align-items:center;
                                        padding:8px 14px;
                                        border-radius:999px;
                                        background:#fff4f4;
                                        color:#ba1a1a;
                                        font-weight:700;
                                        font-size:13px;
                                    "
                                >
                                    Submitted
                                </span>

                            </div>


                            <div
                                style="
                                    display:grid;
                                    grid-template-columns:
                                        repeat(2, minmax(0, 1fr));
                                    gap:18px;
                                    margin-bottom:20px;
                                "
                            >

                                <div>

                                    <small>
                                        Blood group
                                    </small>

                                    <strong
                                        style="
                                            display:block;
                                            margin-top:5px;
                                        "
                                    >
                                        ${esc(
                                            request.bloodGroup
                                        )}
                                    </strong>

                                </div>


                                <div>

                                    <small>
                                        Hospital
                                    </small>

                                    <strong
                                        style="
                                            display:block;
                                            margin-top:5px;
                                        "
                                    >
                                        ${esc(
                                            request.hospital
                                        )}
                                    </strong>

                                </div>


                                <div>

                                    <small>
                                        Location
                                    </small>

                                    <strong
                                        style="
                                            display:block;
                                            margin-top:5px;
                                        "
                                    >
                                        ${esc(
                                            request.location
                                        )}
                                    </strong>

                                </div>


                                <div>

                                    <small>
                                        Created
                                    </small>

                                    <strong
                                        style="
                                            display:block;
                                            margin-top:5px;
                                        "
                                    >
                                        ${esc(date)}
                                    </strong>

                                </div>

                            </div>


                            <button
                                class="secondary"
                                data-route="emergency"
                            >
                                Create another request
                            </button>

                        </div>
                    `;

                })
                .join("");


        app.innerHTML = `
            <section class="page">

                ${pageTop(
                    "My emergency requests",
                    "View the blood requests you have submitted."
                )}

                <div class="donor-list">

                    ${requestCards}

                </div>

            </section>
        `;


        bindCommon();


    } catch (error) {

        console.error(
            "MY REQUESTS ERROR:",
            error
        );

        app.innerHTML = `
            <section class="page">

                ${pageTop(
                    "My emergency requests",
                    "We couldn't load your requests."
                )}

                <div class="panel">

                    <p class="page-subtitle">
                        ${esc(error.message)}
                    </p>

                    <button
                        class="primary"
                        data-route="home"
                    >
                        Back to home
                    </button>

                </div>

            </section>
        `;

        bindCommon();

    }
}

// ============================================================
// RENDER
// ============================================================

async function render() {
    updateHeaderVisibility();

    // Get the current route without the # symbol
    const hash = location.hash.slice(1);

    // --------------------------------------------------
    // PUBLIC DONOR QR PROFILE
    // Does not require login
    // Supports old QR links containing ?format=pdf
    // --------------------------------------------------

    if (hash.startsWith("donor/")) {
        const routePart = hash.slice("donor/".length);
        const encodedToken = routePart.split("?")[0];

        let token;

        try {
            token = decodeURIComponent(encodedToken);
        } catch (error) {
            console.error("Invalid donor QR token:", error);

            app.innerHTML = `
                <section class="page">
                    <h1 class="page-title">Invalid QR Code</h1>
                    <p class="page-subtitle">
                        This donor link is invalid. Please scan the QR code again.
                    </p>
                    <button class="primary" onclick="location.hash = ''">
                        Back Home
                    </button>
                </section>
            `;
            return;
        }

        if (!token) {
            app.innerHTML = `
                <section class="page">
                    <h1 class="page-title">Invalid QR Code</h1>
                    <p class="page-subtitle">
                        The donor token is missing.
                    </p>
                    <button class="primary" onclick="location.hash = ''">
                        Back Home
                    </button>
                </section>
            `;
            return;
        }

        try {
            // Retrieve donor details from Firestore
            const donor = await getPublicDonor(token);

            if (!donor) {
                app.innerHTML = `
                    <section class="page">
                        <p class="eyebrow">SMARTBLOOD PROFILE</p>
                        <h1 class="page-title">Profile Unavailable</h1>
                        <p class="page-subtitle">
                            This donor profile could not be found or may have been removed.
                        </p>
                        <button class="primary" onclick="location.hash = ''">
                            Back Home
                        </button>
                    </section>
                `;
                return;
            }

            // Safely display donor details
            const safe = (value) =>
                typeof esc === "function"
                    ? esc(String(value ?? "Not provided"))
                    : String(value ?? "Not provided");

            app.innerHTML = `
                <section class="page">
                    <p class="eyebrow">SMARTBLOOD PROFILE</p>
                    <h1 class="page-title">Donor Profile</h1>

                    <div class="donor-profile">
                        <p>
                            <strong>Name:</strong>
                            ${safe(donor.name)}
                        </p>

                        <p>
                            <strong>Blood Group:</strong>
                            ${safe(donor.bloodGroup)}
                        </p>

                        <p>
                            <strong>Location:</strong>
                            ${safe(donor.location)}
                        </p>
                    </div>

                    <button class="primary" onclick="location.hash = ''">
                        Back Home
                    </button>
                </section>
            `;
        } catch (error) {
            console.error("Public donor route error:", error);

            app.innerHTML = `
                <section class="page">
                    <h1 class="page-title">Unable to Open Donor Details</h1>
                    <p class="page-subtitle">
                        Unable to load donor information. Please try again.
                    </p>
                    <button class="primary" onclick="location.hash = ''">
                        Back Home
                    </button>
                </section>
            `;
        }

        return;
    }

    // --------------------------------------------------
    // LOGIN PAGE
    // --------------------------------------------------

    if (hash === "login") {
        app.innerHTML = loginPage();
        bind();
        return;
    }

    // --------------------------------------------------
    // SIGNUP PAGE
    // --------------------------------------------------

    if (hash === "signup") {
        app.innerHTML = signupPage();
        bind();
        return;
    }

    // --------------------------------------------------
    // REQUIRE LOGIN FOR PRIVATE PAGES
    // --------------------------------------------------

    if (!auth.currentUser) {
        app.innerHTML = loginPage();
        bind();
        return;
    }

    // --------------------------------------------------
    // NORMAL APPLICATION PAGES
    // --------------------------------------------------

    showLoading();

    try {
        if (hash === "register") {
            app.innerHTML = register();

        } else if (hash === "find") {
            app.innerHTML = find();

        } else if (hash === "availability") {
            app.innerHTML = await availability();

        } else if (hash === "status") {
            await donorStatus();
            return;

        } else if (hash === "requests") {
            await myRequests();
            return;

        } else if (hash === "notifications") {
            app.innerHTML = await notificationsPage();
            bindCommon();
            return;

        } else if (hash === "emergency") {
            app.innerHTML = emergency();

        } else {
            app.innerHTML = await home();
        }

        bind();

    } catch (error) {
        console.error("Page rendering error:", error);

        app.innerHTML = `
            <section class="page">
                <h1 class="page-title">Something Went Wrong</h1>
                <p class="page-subtitle">
                    ${typeof esc === "function"
                        ? esc(error.message || "Please try again.")
                        : "Please try again."}
                </p>
                <button class="primary" onclick="location.hash = ''">
                    Back Home
                </button>
            </section>
        `;

        bindCommon();
    }
}




// ============================================================
// BIND BUTTONS / FORMS
// ============================================================

function bindCommon() {

    document
        .querySelectorAll("[data-route]")
        .forEach(button => {

            button.onclick =
                () => {

                    route(
                        button.dataset.route
                    );

                };

        });

}


function bind() {

    bindCommon();


    // --------------------------------------------------------
    // LOGIN
    // --------------------------------------------------------

    const loginForm =
        document.querySelector(
            "#login-form"
        );
        const forgotPasswordButton =
    document.querySelector(
        "#forgot-password"
    );


if (forgotPasswordButton) {

    forgotPasswordButton.onclick =
        async () => {

            const email =
                document.querySelector(
                    "#login-email"
                ).value.trim();

            await forgotPassword(
                email
            );

        };

}


    if (loginForm) {

        loginForm.addEventListener(
            "submit",
            async event => {

                event.preventDefault();


                const email =
                    document.querySelector(
                        "#login-email"
                    ).value.trim();


                const password =
                    document.querySelector(
                        "#login-password"
                    ).value;


                await loginUser(
                    email,
                    password
                );

            }
        );

    }


    const googleLogin =
        document.querySelector(
            "#google-login"
        );


    if (googleLogin) {

        googleLogin.onclick =
            loginWithGoogle;

    }

    // PHONE OTP LOGIN

    function validateSignupPassword(password) {

    const rules = {

        length:
            password.length >= 8,

        uppercase:
            /[A-Z]/.test(password),

        lowercase:
            /[a-z]/.test(password),

        number:
            /[0-9]/.test(password),

        special:
            /[^A-Za-z0-9]/.test(password)

    };


    const ruleElements = {

        length:
            document.querySelector(
                "#password-length"
            ),

        uppercase:
            document.querySelector(
                "#password-uppercase"
            ),

        lowercase:
            document.querySelector(
                "#password-lowercase"
            ),

        number:
            document.querySelector(
                "#password-number"
            ),

        special:
            document.querySelector(
                "#password-special"
            )

    };


    Object.entries(rules).forEach(
        ([rule, passed]) => {

            const element =
                ruleElements[rule];

            if (!element) {
                return;
            }

            const icon =
                element.querySelector(
                    ".rule-icon"
                );

            if (passed) {

                element.classList.add(
                    "valid"
                );

                if (icon) {
                    icon.textContent = "✓";
                }

            } else {

                element.classList.remove(
                    "valid"
                );

                if (icon) {
                    icon.textContent = "○";
                }

            }

        }
    );


    return Object.values(rules)
        .every(Boolean);
}


    // --------------------------------------------------------
    // SIGNUP
    // --------------------------------------------------------

    const signupForm =
        document.querySelector(
            "#signup-form"
        );
        const signupPassword =
    document.querySelector(
        "#signup-password"
    );


if (signupPassword) {

    signupPassword.addEventListener(
        "input",
        () => {

            validateSignupPassword(
                signupPassword.value
            );

        }
    );

}


    if (signupForm) {

        signupForm.addEventListener(
            "submit",
            async event => {

                event.preventDefault();


                const email =
                    document.querySelector(
                        "#signup-email"
                    ).value.trim();


                const password =
                    document.querySelector(
                        "#signup-password"
                    ).value;


                await createAccount(
                    email,
                    password
                );

            }
        );

    }

// ============================================================
// CONSENT MODAL
// ============================================================

function openConsent(data) {

    modalRoot.innerHTML = `

        <div class="modal-overlay">

            <div class="modal">

                <p class="eyebrow">
                    Before you join
                </p>

                <h2>
                    Consent to be listed
                </h2>

                <p>
                    SmartBlood helps people discover voluntary
                    donors. It does not provide medical advice,
                    verify donor fitness, arrange donations,
                    or guarantee that a donor will be available.
                </p>

                <div class="notice">

                    <b>
                        Important:
                    </b>

                    Do not share sensitive medical information
                    through this directory. Any donation must be
                    assessed and managed by qualified medical staff
                    and the hospital or blood bank.

                </div>

                <label class="checkline">

                    <input
                        type="checkbox"
                        id="consent"
                    />

                    <span>
                        I confirm that my details are accurate,
                        I am voluntarily joining the directory,
                        and I understand SmartBlood is not
                        responsible for medical decisions,
                        donor eligibility, or arrangements
                        between users.
                    </span>

                </label>

                <div class="modal-actions">

                    <button
                        class="secondary"
                        id="cancel-modal"
                        type="button"
                    >
                        Cancel
                    </button>

                    <button
                        class="primary"
                        id="confirm-consent"
                        type="button"
                        disabled
                    >
                        Register me
                    </button>

                </div>

            </div>

        </div>
    `;


    const checkbox =
        document.querySelector("#consent");


    const confirmButton =
        document.querySelector("#confirm-consent");


    checkbox.onchange = event => {

        confirmButton.disabled =
            !event.target.checked;

    };


    document.querySelector(
        "#cancel-modal"
    ).onclick = () => {

        modalRoot.innerHTML = "";

    };


    confirmButton.onclick = async () => {

        confirmButton.disabled = true;

        confirmButton.textContent =
            "Registering...";


        try {

            await saveDonor(data);

        } catch (error) {

            console.error(error);

            toast(
                "Could not register donor. Please try again."
            );

            confirmButton.disabled = false;

            confirmButton.textContent =
                "Register me";

        }

    };

}
   // --------------------------------------------------------
// REGISTER DONOR
// --------------------------------------------------------

const registerForm =
    document.querySelector("#register-form");


if (registerForm) {

    registerForm.addEventListener(
        "submit",
        async event => {

            event.preventDefault();


            const name =
                document.querySelector("#name")
                    ?.value
                    .trim();


            const age =
                Number(
                    document.querySelector("#age")
                        ?.value
                );


            const phone =
                document.querySelector("#phone")
                    ?.value
                    .trim();


            const bloodGroup =
                document.querySelector("#blood")
                    ?.value;


            const location =
                document.querySelector("#location")
                    ?.value
                    .trim();


            const medicalHistory =
                document.querySelector("#medical-history")
                    ?.value
                    .trim();


            // ----------------------------------------
            // VALIDATION
            // ----------------------------------------

            if (
                !name ||
                !age ||
                !phone ||
                !bloodGroup ||
                !location
            ) {

                toast(
                    "Please fill in all required fields."
                );

                return;
            }


            if (
                age < 18 ||
                age > 65
            ) {

                toast(
                    "Donor age must be between 18 and 65."
                );

                return;
            }


            if (
                !/^[0-9]{10}$/.test(phone)
            ) {

                toast(
                    "Please enter a valid 10-digit phone number."
                );

                return;
            }


            // ----------------------------------------
            // OPEN CONSENT POPUP
            // ----------------------------------------

            if (!navigator.geolocation) {
    toast("Location is not supported by this browser.");
    return;
}

navigator.geolocation.getCurrentPosition(
    position => {
        openConsent({
            name,
            age,
            phone,
            bloodGroup,
            location,
            medicalHistory: medicalHistory || "",
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
        });
    },
    error => {
        console.error("Location error:", error);

        if (error.code === 1) {
            toast("Please allow location access to register as a donor.");
        } else {
            toast("Could not get your location. Please try again.");
        }
    },
    {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
    }
);

        }
    );

}


    // --------------------------------------------------------
    // FIND DONORS
    // --------------------------------------------------------

    const searchButton =
        document.querySelector(
            "#search-btn"
        );


    if (searchButton) {

        searchButton.onclick =
            async () => {

                const group =
                    document.querySelector(
                        "#search-group"
                    ).value;


                const resultsBox =
                    document.querySelector(
                        "#search-results"
                    );


                resultsBox.innerHTML = `
                    <p class="empty">
                        Searching donors...
                    </p>
                `;


                try {

                    const results =
                        await searchDonors(
                            group
                        );


                    resultsBox.innerHTML =
                        results.length

                            ? results
                                .map(
                                    donorCard
                                )
                                .join("")

                            : `
                                <p class="empty">
                                    No ${
                                        group || ""
                                    }
                                    donors are listed yet.
                                </p>
                            `;


                    bindProfileButtons();

                } catch (error) {

                    console.error(error);

                    resultsBox.innerHTML = `
                        <p class="empty">
                            Could not load donors.
                            Please try again.
                        </p>
                    `;

                }

            };

    }


    // --------------------------------------------------------
    // EMERGENCY
    // --------------------------------------------------------

    const emergencyForm =
        document.querySelector(
            "#emergency-form"
        );


    if (emergencyForm) {

        emergencyForm.addEventListener(
            "submit",
            async event => {

                event.preventDefault();


                const resultsBox =
                    document.querySelector(
                        "#emergency-results"
                    );


                const patientName =
                    document.querySelector(
                        "#patient"
                    ).value.trim();
                const patientAge =
    Number(
        document.querySelector(
            "#patient-age"
        ).value
    );

const patientMedicalHistory =
    document.querySelector(
        "#patient-medical-history"
    ).value.trim();


                const bloodGroup =
                    document.querySelector(
                        "#required-group"
                    ).value;


                const hospital =
                    document.querySelector(
                        "#hospital"
                    ).value.trim();


                const location =
                    document.querySelector(
                        "#request-location"
                    ).value.trim();


                if (!bloodGroup) {

                    toast(
                        "Please select a blood group."
                    );

                    return;

                }


                resultsBox.innerHTML = `
                    <p class="empty">
                        Finding matching donors...
                    </p>
                `;


                try {

                    const matches =
    await createEmergencyRequest({
        patientName,
        patientAge,
        bloodGroup,
        patientMedicalHistory,
        hospital,
        location
    });


                    resultsBox.innerHTML = `

                        <div
                            class="section-heading"
                            style="margin-top:30px"
                        >

                            <div>

                                <p class="eyebrow">
                                    Request created
                                </p>

                                <h2>
                                    ${
                                        matches.length
                                    }
                                    matching donor${
                                        matches.length === 1
                                            ? ""
                                            : "s"
                                    }
                                    found
                                </h2>

                            </div>

                        </div>

                        ${
                            matches.length

                                ? matches
                                    .map(
                                        donorCard
                                    )
                                    .join("")

                                : `
                                    <p class="empty">
                                        No matching donors are
                                        currently listed.
                                        Contact your hospital
                                        blood bank immediately.
                                    </p>
                                `
                        }
                    `;


                    bindProfileButtons();


                    toast(
                        "Emergency request saved."
                    );

                } catch (error) {

                    console.error(error);

                    resultsBox.innerHTML = `
                        <p class="empty">
                            Could not create the emergency
                            request. Please try again.
                        </p>
                    `;

                }

            }
        );

    }


    // --------------------------------------------------------
    // LOGOUT
    // --------------------------------------------------------

    const logoutButton =
        document.querySelector(
            "#logout-btn"
        );


    if (logoutButton) {

        logoutButton.onclick =
            async () => {

                try {

                    await signOut(
                        auth
                    );

                    toast(
                        "You have been logged out."
                    );

                    route("login");

                } catch (error) {

                    console.error(error);

                    toast(
                        "Logout failed."
                    );

                }

            };

    }


    bindProfileButtons();

}


// ============================================================
// PROFILE BUTTONS
// ============================================================

function bindProfileButtons() {

    document
        .querySelectorAll(
            "[data-profile]"
        )
        .forEach(button => {

            button.onclick =
                () => {

                    const id =
                        button.dataset.profile;

                    privateProfile(id);

                };

        });

}


// ============================================================
// MODAL / TOAST
// ============================================================

function closeModal() {

    modalRoot.innerHTML = "";

}


function toast(message) {

    const root =
        document.querySelector(
            "#toast-root"
        );


    if (!root) return;


    root.innerHTML = `

        <div class="toast">
            ${esc(message)}
        </div>
    `;


    setTimeout(
        () => {

            root.innerHTML = "";

        },
        3200
    );

}



// ============================================================
// HASH CHANGE
// ============================================================

window.addEventListener(
    "hashchange",
    () => {

        render();

    }
);


// ============================================================
// INITIAL LOAD
// ============================================================
render();
// ============================================================
// CREATE EMERGENCY REQUEST
// ============================================================

async function createEmergencyRequest(data) {

    const user =
        auth.currentUser;


    if (!user) {

        route("login");

        return [];

    }


    try {

        // ==========================================
        // 1. GET PATIENT'S CURRENT LOCATION
        // ==========================================

        let latitude = null;
        let longitude = null;


        if (navigator.geolocation) {

            try {

                const position =
                    await new Promise(
                        (resolve, reject) => {

                            navigator.geolocation.getCurrentPosition(
                                resolve,
                                reject,
                                {
                                    enableHighAccuracy: true,
                                    timeout: 10000,
                                    maximumAge: 0
                                }
                            );

                        }
                    );


                latitude =
                    position.coords.latitude;

                longitude =
                    position.coords.longitude;


                console.log(
                    "📍 Patient coordinates:",
                    latitude,
                    longitude
                );

            } catch (locationError) {

                console.error(
                    "Could not get patient location:",
                    locationError
                );

                toast(
                    "Could not get your exact location. Donors will still be searched."
                );

            }

        }


        // ==========================================
        // 2. CREATE EMERGENCY BLOOD REQUEST
        // ==========================================

        const requestRef =
            await addDoc(
                collection(
                    db,
                    "bloodRequests"
                ),
                {

                    patientName:
                        data.patientName,

                    patientAge:
                        data.patientAge,

                    bloodGroup:
                        data.bloodGroup,

                    patientMedicalHistory:
                        data.patientMedicalHistory || "",

                    hospital:
                        data.hospital,

                    location:
                        data.location,

                    latitude:
                        latitude,

                    longitude:
                        longitude,

                    status:
                        "open",

                    createdBy:
                        user.uid,

                    timestamp:
                        serverTimestamp()

                }
            );


        console.log(
            "🩸 Emergency request created:",
            requestRef.id
        );


        // ==========================================
        // 3. FIND MATCHING AVAILABLE DONORS
        // ==========================================

        const q =
            query(
                collection(
                    db,
                    "donors"
                ),

                where(
                    "bloodGroup",
                    "==",
                    data.bloodGroup
                ),

                where(
                    "active",
                    "==",
                    true
                )
            );


        const snapshot =
            await getDocs(q);


        const donors =
            snapshot.docs
                .map(document => ({

                    id:
                        document.id,

                    ...document.data()

                }))
                .filter(
                    donor =>
                        donor.availability !==
                        "unavailable"
                );


        console.log(
            "🩸 Matching donors:",
            donors.length
        );


        // ==========================================
        // 4. CALCULATE DISTANCE FROM PATIENT
        // ==========================================

        const donorsWithDistance =
            donors.map(
                donor => {

                    if (
                        latitude === null ||
                        longitude === null ||
                        typeof donor.latitude !== "number" ||
                        typeof donor.longitude !== "number"
                    ) {

                        return {

                            ...donor,

                            distanceKm:
                                null

                        };

                    }


                    const distance =
                        calculateDistance(
                            latitude,
                            longitude,
                            donor.latitude,
                            donor.longitude
                        );


                    return {

                        ...donor,

                        distanceKm:
                            distance

                    };

                }
            );


        // ==========================================
        // 5. SORT NEAREST DONORS FIRST
        // ==========================================

        donorsWithDistance.sort(
            (a, b) => {

                if (
                    a.distanceKm === null
                ) {

                    return 1;

                }


                if (
                    b.distanceKm === null
                ) {

                    return -1;

                }


                return (
                    a.distanceKm -
                    b.distanceKm
                );

            }
        );


        // ==========================================
        // 6. CREATE NOTIFICATION FOR EACH DONOR
        // ==========================================

        await Promise.all(

            donorsWithDistance.map(
                async donor => {

                    if (!donor.ownerUid) {

                        console.warn(
                            "Donor has no ownerUid:",
                            donor.id
                        );

                        return;

                    }


                    let distanceText =
                        "";


                    if (
                        typeof donor.distanceKm ===
                        "number"
                    ) {

                        distanceText =
                            donor.distanceKm < 1

                                ? `${Math.round(
                                    donor.distanceKm * 1000
                                )} m away`

                                : `${donor.distanceKm.toFixed(
                                    1
                                )} km away`;

                    }


                    await addDoc(
                        collection(
                            db,
                            "notifications"
                        ),
                        {

                            // Donor who receives notification
                            recipientUid:
                                donor.ownerUid,

                            // Emergency request ID
                            requestId:
                                requestRef.id,

                            // Patient/request creator
                            requestCreatedBy:
                                user.uid,

                            title:
                                "Emergency blood request",

                            message:
                                `${data.bloodGroup} blood is urgently required at ${data.hospital || "the requested hospital"}.`,

                            patientName:
                                data.patientName,

                            patientAge:
                                data.patientAge,

                            bloodGroup:
                                data.bloodGroup,

                            hospital:
                                data.hospital,

                            location:
                                data.location,

                            distanceKm:
                                donor.distanceKm,

                            distanceText:
                                distanceText,

                            read:
                                false,

                            createdAt:
                                serverTimestamp()

                        }
                    );


                    console.log(
                        "🔔 Notification created for donor:",
                        donor.name
                    );

                }
            )

        );


        console.log(
            "🔔 All donor notifications created."
        );


        // ==========================================
        // 7. RETURN DONORS TO EMERGENCY PAGE
        // ==========================================

        return donorsWithDistance;


    } catch (error) {

        console.error(
            "Could not create emergency request:",
            error
        );

        throw error;

    }

}
// ============================================================
// GET OPEN EMERGENCY REQUESTS FOR DONOR
// ============================================================

async function getOpenEmergencyRequestsForDonor() {

    const user =
        auth.currentUser;


    if (!user) {
        return [];
    }


    try {

        const donor =
            await getMyDonorRecord();


        if (!donor) {
            return [];
        }


        const q =
            query(
                collection(
                    db,
                    "bloodRequests"
                ),

                where(
                    "bloodGroup",
                    "==",
                    donor.bloodGroup
                )
            );


        const snapshot =
            await getDocs(q);


        return snapshot.docs
            .map(document => ({
                id: document.id,
                ...document.data()
            }))
            .filter(
                request =>
                    request.status === "open" ||
                    request.status === "accepted"
            );


    } catch (error) {

        console.error(
            "Could not load emergency requests:",
            error
        );

        return [];

    }
}
// ============================================================
// MY EMERGENCY REQUESTS
// ============================================================

async function getMyEmergencyRequests() {

    const user =
        auth.currentUser;


    if (!user) {
        return [];
    }


    try {

        const q =
            query(
                collection(
                    db,
                    "bloodRequests"
                ),

                where(
                    "createdBy",
                    "==",
                    user.uid
                )
            );


        const snapshot =
            await getDocs(q);


        return snapshot.docs
            .map(document => ({

                id:
                    document.id,

                ...document.data()

            }));


    } catch (error) {

        console.error(
            "Could not load my emergency requests:",
            error
        );

        return [];

    }
}
// ============================================================
// MY EMERGENCY REQUESTS PAGE
// ============================================================

async function myEmergencyRequests() {

    const requests =
        await getMyEmergencyRequests();


    return `
        <section class="page">

            ${pageTop(
                "My emergency requests",
                "Track your blood requests and see when a donor accepts."
            )}


            ${
                requests.length === 0

                    ? `

                        <div class="panel">

                            <h2>
                                No emergency requests
                            </h2>

                            <p class="page-subtitle">
                                You have not created any emergency
                                blood requests yet.
                            </p>

                        </div>

                    `

                    : `

                        <div class="donor-list">

                            ${requests.map(request => `

                                <article
                                    class="panel"
                                    data-request-id="${request.id}"
                                >

                                    <p class="eyebrow">
                                        ${esc(request.bloodGroup)}
                                        blood required
                                    </p>


                                    <h2>
                                        ${esc(request.patientName)}
                                    </h2>


                                    <p>
                                        <strong>
                                            Patient age:
                                        </strong>

                                        ${
                                            request.patientAge ??
                                            "Not provided"
                                        }
                                    </p>


                                    <p>
                                        <strong>
                                            Hospital:
                                        </strong>

                                        ${esc(request.hospital)}
                                    </p>


                                    <p>
                                        <strong>
                                            Location:
                                        </strong>

                                        ${esc(request.location)}
                                    </p>


                                    <p class="page-subtitle">

                                        Status:

                                        <strong>
                                            ${esc(
                                                request.status ||
                                                "open"
                                            )}
                                        </strong>

                                    </p>


                                    ${
                                        request.status === "accepted"

                                            ? `

                                                <div class="panel">

                                                    <p class="eyebrow">
                                                        DONOR ACCEPTED
                                                    </p>


                                                    <h3>
                                                        ${esc(
                                                            request.responseDonorName ||
                                                            "Donor"
                                                        )}
                                                    </h3>


                                                    <p>
                                                        <strong>
                                                            Blood group:
                                                        </strong>

                                                        ${esc(
                                                            request.responseDonorBloodGroup ||
                                                            request.bloodGroup
                                                        )}
                                                    </p>


                                                    <p>
                                                        <strong>
                                                            Phone:
                                                        </strong>

                                                        ${esc(
                                                            request.responseDonorPhone ||
                                                            "Not available"
                                                        )}
                                                    </p>


                                                    <p>
                                                        <strong>
                                                            Location:
                                                        </strong>

                                                        ${esc(
                                                            request.responseDonorLocation ||
                                                            "Not available"
                                                        )}
                                                    </p>


                                                    ${
                                                        request.responseDonorPhone

                                                            ? `

                                                                <a
                                                                    class="primary"
                                                                    href="tel:${request.responseDonorPhone}"
                                                                >
                                                                    Contact donor
                                                                </a>

                                                            `

                                                            : ""
                                                    }

                                                </div>

                                            `

                                            : request.status === "declined"

                                                ? `

                                                    <div class="notice">

                                                        This request was
                                                        declined by the donor.

                                                    </div>

                                                `

                                                : `

                                                    <div class="notice">

                                                        Waiting for a matching
                                                        donor to respond.

                                                    </div>

                                                `
                                    }

                                </article>

                            `).join("")}

                        </div>

                    `
            }

        </section>
    `;
}
// ============================================================
// DONOR EMERGENCY REQUESTS
// ============================================================

async function donorEmergencyRequests() {

    const requests =
        await getOpenEmergencyRequestsForDonor();


    return `
        <section class="page">

            ${pageTop(
                "Emergency requests",
                "View open blood requests matching your blood group."
            )}


            ${
                requests.length === 0

                    ? `

                        <div class="panel">

                            <h2>
                                No open requests
                            </h2>

                            <p class="page-subtitle">
                                There are currently no emergency
                                requests matching your blood group.
                            </p>

                        </div>

                    `

                    : `

                        <div class="donor-list">

                            ${requests.map(request => `

                                <article
    class="panel"
    data-request-id="${request.id}"
>

                                    <p class="eyebrow">
                                        ${request.bloodGroup}
                                        blood required
                                    </p>


                                    <h2>
                                        ${request.patientName}
                                    </h2>


                                    <p>
                                        <strong>
                                            Patient age:
                                        </strong>

                                        ${request.patientAge ?? "Not provided"}
                                    </p>


                                    <p>
                                        <strong>
                                            Hospital:
                                        </strong>

                                        ${request.hospital}
                                    </p>


                                    <p>
                                        <strong>
                                            Location:
                                        </strong>

                                        ${request.location}
                                    </p>


                                    ${
                                        request.patientMedicalHistory

                                            ? `

                                                <p>
                                                    <strong>
                                                        Medical history:
                                                    </strong>

                                                    ${request.patientMedicalHistory}
                                                </p>

                                            `

                                            : ""
                                    }


                                    <p class="page-subtitle">
                                        Status:
                                        <strong>
                                            ${request.status}
                                        </strong>
                                    </p>


                                    <div class="modal-actions">

    ${
        request.status === "accepted"

            ? `
                <button
                    class="primary"
                    type="button"
                    disabled
                >
                    ✓ Accepted
                </button>
            `

            : `
                <button
                    class="primary request-action"
                    type="button"
                    data-request-id="${request.id}"
                    data-action="accept"
                >
                    Accept
                </button>


                <button
                    class="secondary request-action"
                    type="button"
                    data-request-id="${request.id}"
                    data-action="decline"
                >
                    Decline
                </button>
            `
    }

</div>

                                </article>

                            `).join("")}

                        </div>

                    `
            }

        </section>
    `;
}
document.addEventListener(
    "click",
    async event => {

        const button =
            event.target.closest(
                '[data-action="logout"]'
            );

        if (!button) {
            return;
        }

        try {

            button.disabled = true;
            button.textContent = "Logging out...";

            await signOut(auth);

            location.hash = "login";

            await render();

            toast(
                "You have been logged out."
            );

        } catch (error) {

            console.error(
                "Logout failed:",
                error
            );

            button.disabled = false;
            button.textContent = "Logout";

            toast(
                "Could not log out. Please try again."
            );

        }

    }
);

/// ============================================================
// EMERGENCY REQUEST BUTTONS
// ============================================================
document.addEventListener(
    "click",
    async event => {

        const button =
            event.target.closest(
                ".notification-request-button"
            );

        if (!button) {
            return;
        }


        const requestId =
            button.dataset.requestId;


        if (!requestId) {

            toast(
                "Emergency request could not be found."
            );

            return;

        }


        const user =
            auth.currentUser;


        if (!user) {

            route("login");

            return;

        }


        try {

            const requestSnapshot =
                await getDoc(
                    doc(
                        db,
                        "bloodRequests",
                        requestId
                    )
                );


            if (!requestSnapshot.exists()) {

                toast(
                    "This emergency request no longer exists."
                );

                return;

            }


            const request = {
                id: requestSnapshot.id,
                ...requestSnapshot.data()
            };


            app.innerHTML = `
                <section class="page">

                    ${pageTop(
                        "Emergency blood request",
                        "Review the request and respond if you can help."
                    )}

                    <article class="panel">

                        <p class="eyebrow">
                            ${esc(request.bloodGroup)}
                            blood required
                        </p>

                        <h2>
                            ${esc(request.patientName)}
                        </h2>

                        <p>
                            <strong>
                                Patient age:
                            </strong>
                            ${
                                request.patientAge ??
                                "Not provided"
                            }
                        </p>

                        <p>
                            <strong>
                                Hospital:
                            </strong>
                            ${esc(
                                request.hospital ||
                                "Not provided"
                            )}
                        </p>

                        <p>
                            <strong>
                                Location:
                            </strong>
                            ${esc(
                                request.location ||
                                "Not provided"
                            )}
                        </p>

                        <p>
                            <strong>
                                Status:
                            </strong>
                            ${esc(
                                request.status ||
                                "open"
                            )}
                        </p>

                        <div class="notice">

                            <strong>
                                Medical history:
                            </strong>

                            <p>
                                ${
                                    esc(
                                        request.patientMedicalHistory ||
                                        "Not provided"
                                    )
                                }
                            </p>

                        </div>


                        ${
                            request.status === "open"
                                ? `
                                    <div class="modal-actions">

                                        <button
                                            class="primary request-action"
                                            type="button"
                                            data-request-id="${esc(
                                                request.id
                                            )}"
                                            data-action="accept"
                                        >
                                            Accept
                                        </button>

                                        <button
                                            class="secondary request-action"
                                            type="button"
                                            data-request-id="${esc(
                                                request.id
                                            )}"
                                            data-action="decline"
                                        >
                                            Decline
                                        </button>

                                    </div>
                                `
                                : `
                                    <div class="notice">
                                        This request is already
                                        ${esc(request.status)}.
                                    </div>
                                `
                        }

                    </article>

                </section>
            `;


            bindCommon();

        } catch (error) {

            console.error(
                "Could not load emergency request:",
                error
            );

            toast(
                "Could not load the emergency request."
            );

        }

    }
);
// ============================================================
// ACCEPT / DECLINE EMERGENCY REQUEST
// ============================================================

document.addEventListener(
    "click",
    async event => {

        const button =
            event.target.closest(
                ".request-action"
            );


        if (!button) {
            return;
        }


        const requestId =
            button.dataset.requestId;


        const action =
            button.dataset.action;


        if (!requestId || !action) {
            return;
        }


        const user =
            auth.currentUser;


        if (!user) {

            route("login");

            return;
        }


        // Prevent double clicking
        button.disabled = true;


        const originalText =
            button.textContent;


        button.textContent =
            action === "accept"
                ? "Accepting..."
                : "Declining...";


        try {

            const newStatus =
                action === "accept"
                    ? "accepted"
                    : "declined";


            const donor =
    await getMyDonorRecord();


await updateDoc(
    doc(
        db,
        "bloodRequests",
        requestId
    ),
    {

        status:
            newStatus,

        responseBy:
            user.uid,

        responseTimestamp:
            serverTimestamp(),

        responseDonorName:
            donor?.name || "",

        responseDonorPhone:
            donor?.phone || "",

        responseDonorBloodGroup:
            donor?.bloodGroup || "",

        responseDonorLocation:
            donor?.location || ""

    }
);


           toast(
    action === "accept"
        ? "Emergency request accepted."
        : "Emergency request declined."
);


// Reload the emergency requests page
route("requests");


        } catch (error) {

            console.error(
                "Could not update emergency request:",
                error
            );


            toast(
                "Could not update the request. Please try again."
            );


            button.disabled = false;

            button.textContent =
                originalText;

        }

    }
);
// ============================================================
// GLOBAL PAGE NAVIGATION
// ============================================================

document.addEventListener(
    "click",
    event => {

        const button =
            event.target.closest(
                "[data-route]"
            );


        if (!button) {
            return;
        }


        event.preventDefault();


        const destination =
            button.dataset.route;


        if (!destination) {
            return;
        }


        route(destination);

    }
);
document.addEventListener(
    "click",
    function (event) {

        const button =
            event.target.closest(
                ".password-toggle"
            );

        if (!button) {
            return;
        }

        event.preventDefault();

        const inputId =
            button.getAttribute(
                "data-password-toggle"
            );

        const input =
            document.getElementById(
                inputId
            );

        if (!input) {
            return;
        }

        if (input.type === "password") {

            input.type = "text";

            button.textContent = "🙈";

            button.setAttribute(
                "aria-label",
                "Hide password"
            );

        } else {

            input.type = "password";

            button.textContent = "👁";

            button.setAttribute(
                "aria-label",
                "Show password"
            );

        }

    }
);
if ("serviceWorker" in navigator) {

    window.addEventListener("load", () => {

        navigator.serviceWorker
            .register("/service-worker.js")
            .then(() => {
                console.log(
                    "e-SmartBlood service worker registered."
                );
            })
            .catch(error => {
                console.error(
                    "Service worker registration failed:",
                    error
                );
            });

    });

}
/* =========================================================
   MOBILE NAVIGATION
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    const mobileMenuButton =
        document.getElementById("mobileMenuButton");

    const siteNav =
        document.getElementById("siteNav");

    if (!mobileMenuButton || !siteNav) {
        return;
    }


    /* Open / close menu */

    mobileMenuButton.addEventListener("click", (event) => {

        event.stopPropagation();

        const isOpen =
            siteNav.classList.contains("mobile-nav-open");

        if (isOpen) {

            siteNav.classList.remove(
                "mobile-nav-open"
            );

            mobileMenuButton.textContent = "☰";

            mobileMenuButton.setAttribute(
                "aria-expanded",
                "false"
            );

            mobileMenuButton.setAttribute(
                "aria-label",
                "Open menu"
            );

        } else {

            siteNav.classList.add(
                "mobile-nav-open"
            );

            mobileMenuButton.textContent = "✕";

            mobileMenuButton.setAttribute(
                "aria-expanded",
                "true"
            );

            mobileMenuButton.setAttribute(
                "aria-label",
                "Close menu"
            );
        }

    });


    /* Close menu after selecting any navigation option */

    siteNav.addEventListener("click", (event) => {

        const button =
            event.target.closest("button");

        if (!button) {
            return;
        }

        siteNav.classList.remove(
            "mobile-nav-open"
        );

        mobileMenuButton.textContent = "☰";

        mobileMenuButton.setAttribute(
            "aria-expanded",
            "false"
        );

        mobileMenuButton.setAttribute(
            "aria-label",
            "Open menu"
        );

    });


    /* Close when tapping outside */

    document.addEventListener("click", (event) => {

        if (
            !siteNav.contains(event.target) &&
            !mobileMenuButton.contains(event.target)
        ) {

            siteNav.classList.remove(
                "mobile-nav-open"
            );

            mobileMenuButton.textContent = "☰";

            mobileMenuButton.setAttribute(
                "aria-expanded",
                "false"
            );

            mobileMenuButton.setAttribute(
                "aria-label",
                "Open menu"
            );
        }

    });


    /* If screen becomes desktop, reset mobile menu */

    window.addEventListener("resize", () => {

        if (window.innerWidth > 768) {

            siteNav.classList.remove(
                "mobile-nav-open"
            );

            mobileMenuButton.textContent = "☰";

            mobileMenuButton.setAttribute(
                "aria-expanded",
                "false"
            );
        }

    });

});
