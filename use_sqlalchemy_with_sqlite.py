#!/usr/bin/env python3
from sqlalchemy import create_engine, event
from sqlalchemy import text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import TypeDecorator, String
from sqlalchemy.ext.asyncio import create_async_engine

# Create an engine connected to the SQLite database
engine_01 = create_engine("sqlite:///sports_cars_tutorial.db")

# Define and create tables with SQLAlchemy
Base = declarative_base()

class Sportscar(Base):
    __tablename__ = "sportscar"
    id = Column(Integer, primary_key=True)
    make = Column(String)
    model = Column(String)
    year = Column(Integer)
    horsepower = Column(Integer)

# Create all tables in the engine
Base.metadata.create_all(engine_01)

# Create session for interacting with SQLAlchemy
Session = sessionmaker(bind=engine_01)
session = Session()

# Insert records into the sportscar table
ferrari = Sportscar(make="Ferrari", model="F8 Tributo", year=2020, horsepower=710)
lamborghini = Sportscar(make="Lamborghini", model="Huracan EVO", year=2020, horsepower=640)

session.add(ferrari)
session.add(lamborghini)

# Commit the transaction
session.commit()

# Query for all sportscars
for sportscar in session.query(Sportscar).all():
    print(sportscar.make, sportscar.model)

# Inserting multiple records with SQLAlchemy
data = [
    Sportscar(make="Porsche", model="911 Turbo S", year=2020, horsepower=640),
    Sportscar(make="McLaren", model="720S", year=2020, horsepower=710),
    Sportscar(make="Aston Martin", model="DBS Superleggera", year=2020, horsepower=715)
]

session.add_all(data)
session.commit()


class Colour:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

# Assuming a Colour class is defined elsewhere
class ColourType(TypeDecorator):
    impl = String
    def process_bind_param(self, value, dialect):
        return f"{value.r}, {value.g}, {value.b}" if value else None

    def process_result_value(self, value, dialect):
        r, g, b = map(int, value.split(", "))
        return Colour(r, g, b)

# Create another engine with SQLcipher to the SQLite database
engine_02 = create_engine("sqlite+pysqlcipher://:passphrase@/encrypted_database.db")

# Creating an async engine with SQLAlchemy and aiosqlite
async_engine = create_async_engine("sqlite+aiosqlite:///async_database.db")

# User-defined function (UDFs) and advanced SQLite usage
def my_custom_function(x, y):
    return x * y

engine_03 = create_engine("sqlite:///engine_03_db.db")

@event.listens_for(engine_03, "connect")
def connect(dbapi_connection, connection_record):
    dbapi_connection.create_function("my_custom_function", 2, my_custom_function)

engine_04 = create_engine("sqlite:///engine_04_db.db")

@event.listens_for(engine_04, "connect")
def do_connect(dbapi_connection, connection_record):
    dbapi_connection.isolation_level = None

@event.listens_for(engine_04, "begin")
def do_begin(conn):
    conn.exec_driver_sql("BEGIN DEFERRED TRANSACTION")

