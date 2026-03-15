from App.api.databases.Tables import UserPersona
from App.api.dependencies.sqlite_connector import SessionLocal

db = SessionLocal()
assignment = UserPersona(user_id=2, persona_id=1)
db.add(assignment)
db.commit()

existing = db.query(UserPersona).filter(
    UserPersona.user_id == 2,
    UserPersona.persona_id == 1
).first()

print(f"User binding exists: {existing is not None}, id={existing.id}")
