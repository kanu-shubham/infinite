import React from "react";
import "./UserProfileSelector.css";

export default function UserProfileSelector({ profiles, selectedId, onSelect }) {
  return (
    <section className="profile-selector">
      <div className="profile-selector__header">
        <h2 className="profile-selector__title">Who is traveling?</h2>
        <p className="profile-selector__subtitle">
          Switch profiles to see how recommendations change for different traveler types
        </p>
      </div>
      <div className="profile-selector__list">
        {profiles.map((profile) => {
          const isActive = profile.id === selectedId;
          return (
            <button
              key={profile.id}
              className={`profile-card${isActive ? " profile-card--active" : ""}`}
              onClick={() => onSelect(profile.id)}
              aria-pressed={isActive}
            >
              <div className="profile-card__avatar">{profile.avatar}</div>
              <div className="profile-card__info">
                <span className="profile-card__name">{profile.name}</span>
                <span className="profile-card__tagline">{profile.tagline}</span>
              </div>
              <div className="profile-card__trip">
                <span className="profile-card__destination">
                  {profile.upcomingTrip.destination}
                </span>
                <span className="profile-card__duration">
                  {profile.upcomingTrip.durationDays}d
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
